import torch
import config
import time
import numpy as np
from threading import Lock
from simulators.simulator import Simulator
from dynamics_identification.torch_dynamics_models.single_track_bicycle import SingleTrackBicycle
from dynamics_identification.torch_dynamics_models.gaussian_process import GPModel


'''
Implements a vehicle simulation using the single track bicycle model ODEs,

The simulation is stepped forward only when a control is applied from the control queue, 
which allows the controller to dictate the flow speed of the simulation.

'''
class CasadiSimulator(Simulator):
    def __init__(self, shared_car_state, controls_queue, track, track_name, initial_max_speed):
        super().__init__(controls_queue, track, track_name, initial_max_speed)
        self.shared_car_state = shared_car_state

        self.dynamic_bicycle = SingleTrackBicycle(sim_mode=True)

        self.state_lock = Lock()
        self.state = np.zeros(8)
        self.solve_time = 0
        self.dstate = np.zeros(3)

        self.time_passed = 0

    def reset(self):
        self.state_lock.acquire()
        self.state = np.zeros(8)
        self.state[:2] = self.interpolated_track[100, 1:3]
        self.state[2] = self.interpolated_track[100, -1]
        self.state_lock.release()

    def read_state(self):
        timestamp = time.time_ns()
        done_cause = None
        if self.done_event.is_set():
            done_cause = "requested by controller"

        if self.start_time is None:
            self.start_time = timestamp

        with self.state_lock:
            track_pos = self.get_track_pos(self.state[:2], self.time_passed)
            if len(self.lap_times) > config.num_eval_laps:
                done_cause = "all laps done"

            car_state = np.r_[
                self.time_passed,
                self.state[:-2],    # x, y, hdg, vx, vy, w
                self.dstate,        # ax, ay, dw
                self.state[6:8],    # steer, throttle
                track_pos,
                self.max_speed
            ]

        self.shared_car_state.set_state(car_state[:14])
        self._store_state(car_state[:14])

        return car_state[:13], self.solve_time, done_cause

    def _set_controls(self, steer, throttle, d_steer, solve_time):
        with self.state_lock:
            self.time_passed += config.mpc_sample_time
            self.state[6] = -steer
            self.state[7] = throttle

            input_state = torch.from_numpy(self.state).to(torch.float).unsqueeze(0)
            with torch.no_grad():
                # Runge kutta 4:
                k1 = self.dynamic_bicycle(input_state[:, 2:])
                k2 = self.dynamic_bicycle((input_state + config.mpc_sample_time / 2 * k1)[:, 2:])
                k3 = self.dynamic_bicycle((input_state + config.mpc_sample_time / 2 * k2)[:, 2:])
                k4 = self.dynamic_bicycle((input_state + config.mpc_sample_time * k3)[:, 2:])
                state_delta = config.mpc_sample_time / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                self.state[:6] += state_delta.squeeze(0).numpy()[:6]

                vx = input_state[0, 3]
                vy = input_state[0, 4]
                w = input_state[0, 5]
                undo_sim = torch.tensor([-vy*w, vx*w, 0])
                self.dstate = k1[0, 3:6] + undo_sim

            self.solve_time = solve_time