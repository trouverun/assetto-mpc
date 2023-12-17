import time
import config
import argparse
import sys
import numpy as np
from multiprocessing import Queue, Process, Event, Lock, Barrier
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from controllers.acados_controller import AcadosController
from controllers.manual_controller import ManualController
from simulators.assetto_simulator.assetto_simulator import AssettoSimulator
from simulators.torch_simulator.torch_simulator import CasadiSimulator
from utils.learning_storage import LearningStorage, MockLearningStorage
from odometry_visualizer import Visualizer
from utils.general import CarStateSmWrapper
from PyQt6 import QtWidgets


'''
Reads the shared memory associated with the car state and feeds it to the controller.

The controller sends a state- and control horizon back to the simulation through a queue, which is used to control the car
and visualize the scenario.

'''
def control_loop(car_state_memory_name, car_state_lock, sim_ready_barrier, sim_type, controller_type, track, reference,
                 horizon_queue, sim_que, controller_learning_queue, exit_event):
    car_state_memory = SharedMemory(car_state_memory_name)
    shared_car_state = CarStateSmWrapper(car_state_memory, car_state_lock)

    if controller_type in ["MPC", "GP-MPC", "UCA-MPC"]:
        controller = AcadosController(controller_learning_queue,
                                      delay_compensation=sim_type == "assetto",
                                      use_gp=controller_type in ["GP-MPC", "UCA-MPC"],
                                      constraint_tightening=controller_type == "UCA-MPC"
                                      )
    else:
        controller = ManualController()
    controller.initialize(track, reference)

    sim_ready_barrier.wait()
    time.sleep(0.1)

    while True:
        t1 = time.time_ns()

        if exit_event.is_set():
            controller.kill()
            car_state_memory.close()
            return

        initial_state = shared_car_state.controller_state
        state_horizon, control_horizon, track_tighteners, done_cause = controller.get_control(
            initial_state, shared_car_state.max_speed)

        t2 = time.time_ns()
        horizon_queue.put(state_horizon)
        sim_que.put((state_horizon, control_horizon, track_tighteners, done_cause, (t2-t1)/1e6, time.time_ns()))
        time_passed_s = (t2-t1)/1e9
        time.sleep(max(0, config.mpc_sample_time-time_passed_s))


'''
Manages the simulation handle and logs data from the simulation. 

The purpose of this loop is to produce data at a higher rate than what the control loop would provide (~20Hz vs >100Hz).

When using acados controller, the data storage is also used to formulate a filtered version of states (and derivatives),
which are sent to the acados controller for learning the system dynamics.

'''
def main_loop(shared_car_state, sim_ready_barrier, sim_type, controller_type, track, track_name, experiment,
              sim_que, odometry_que, controller_learning_queue):
    if experiment == "ideal":
        initial_max_speed = config.speed_limit
    else:
        initial_max_speed = config.minimum_speed

    if sim_type == "assetto":
        data_storage = LearningStorage()
        simulator = AssettoSimulator(shared_car_state, sim_que, track, track_name, initial_max_speed)
    else:
        data_storage = MockLearningStorage()  # Pointless to store here
        simulator = CasadiSimulator(shared_car_state, sim_que, track, track_name, initial_max_speed)

    simulator.reset()

    sim_ready_barrier.wait()

    iters_till_learning = 0
    while True:
        t1 = time.time_ns()

        try:
            car_state, solve_time, done_cause = simulator.read_state()
            if done_cause is not None:
                print("Done cause: %s" % done_cause)
                simulator.close(sim_type)
                break
        except ConnectionRefusedError:
            continue
        odometry_que.put((car_state, solve_time))

        if controller_type in ["GP-MPC", "UCA-MPC"]:
            pos_info, learning_inputs, learning_outputs = data_storage.get_real_time_learning_sample(car_state)
            if learning_inputs is not None:
                if iters_till_learning == 0:
                    controller_learning_queue.put((pos_info, learning_inputs, learning_outputs))
                    iters_till_learning = int(config.learning_delay / config.data_sample_time)
                else:
                    iters_till_learning -= 1

        t2 = time.time_ns()
        time_passed_s = (t2-t1)/1e9
        time.sleep(max(0, config.data_sample_time-time_passed_s))


def launch_odometry_window(odometry_que, horizon_queue, exit_event):
    app = QtWidgets.QApplication(sys.argv)
    vis = Visualizer(odometry_que, horizon_queue, exit_event)
    vis.show()
    app.exec()  # Blocks until window is closed
    if not exit_event.is_set():
        exit_event.set()


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', type=str)
    parser.add_argument('--controller', type=str)
    parser.add_argument('--track', type=str, required=False)
    parser.add_argument('--experiment', type=str, default="ideal")
    args = parser.parse_args()

    if args.controller not in ["MPC", "GP-MPC", "UCA-MPC", "manual"]:
        raise ValueError("Unknown controller")

    if args.sim not in ["torch", "assetto"]:
        raise ValueError("Unknown sim")

    if args.experiment not in ["ideal", "challenging"]:
        raise ValueError("Unknown experiment")

    # When driving manually to collect data, track file is not needed:
    try:
        track = np.load("tracks/%s.npy" % args.track)
        track_name = args.track
    except:
        raise ValueError("Couldn't load track %s" % args.track)

    try:
        reference = np.load(f"tracks/{args.track}_trajectory_test.npy")
    except:
        raise ValueError("Couldn't load track %s" % args.track)

    # The processes communicate using a numpy array backed by shared memory, initialize along with the associated lock:
    with SharedMemoryManager() as smm:
        car_state_lock = Lock()
        n_bytes = CarStateSmWrapper.N_BYTES
        car_state_memory = smm.SharedMemory(size=n_bytes)
        shared_car_state = CarStateSmWrapper(car_state_memory, car_state_lock)

        odometry_queue = Queue()    # For the pyqtgraph state visualization
        horizon_queue = Queue()     # For drawing the 2d pygame scenario
        sim_queue = Queue()         # Controls for the simulation
        controller_learning_queue = Queue()
        exit_event = Event()

        # Launch the pyqtgraph window in another process:
        odometry_process = Process(target=launch_odometry_window, args=(odometry_queue, horizon_queue, exit_event))
        odometry_process.start()

        sim_ready_barrier = Barrier(2)
        control_process = Process(target=control_loop, args=(
            car_state_memory.name, car_state_lock, sim_ready_barrier, args.sim, args.controller, track, reference,
            horizon_queue, sim_queue, controller_learning_queue, exit_event))
        control_process.start()

        # Needed for manual controller, otherwise the emulated controller activates before the real one
        time.sleep(1)

        main_loop(shared_car_state, sim_ready_barrier, args.sim, args.controller, track, track_name, args.experiment,
                  sim_queue, odometry_queue, controller_learning_queue)

        exit_event.set()
        time.sleep(1)
        control_process.kill()
        print("Trying to join control process")
        control_process.join()
        print("trying to kill odometry process")
        odometry_process.kill()
        print("trying to join odometry process")
        odometry_process.join()


if __name__ == "__main__":
    setup()