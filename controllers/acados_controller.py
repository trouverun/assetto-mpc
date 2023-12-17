import json
import grpc
import torch
import config
import numpy as np
from controllers.acados_interface.mpc_pb2_grpc import ModelPredictiveControllerStub
from controllers.acados_interface.mpc_pb2 import Settings, Problem, LearningData, Empty
from controllers.controller import Controller
from threading import Thread
from dynamics_identification.torch_dynamics_models.single_track_bicycle import SingleTrackBicycle


class AcadosController(Controller):
    def __init__(self, learning_queue, delay_compensation=True, use_gp=False, constraint_tightening=False):
        self.learning_queue = learning_queue
        self.delay_compensation = delay_compensation
        self.use_gp=use_gp
        self.constraint_tightening=constraint_tightening

        options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]
        channel = grpc.insecure_channel("localhost:8000", options)
        self.stub = ModelPredictiveControllerStub(channel)
        self.failed_solves = 0
        self.learning_thread = Thread(target=self._learning_fn, args=())
        self.learning_thread.start()

    def _learning_fn(self):
        bicycle_model = SingleTrackBicycle(sim_mode=False)
        while True:
            pos_info, inputs, outputs = self.learning_queue.get()
            if inputs is None:
                break

            if inputs[0] < 20:
                continue

            torch_inputs = torch.from_numpy(np.r_[0, inputs, np.zeros(3)]).unsqueeze(0)
            with torch.no_grad():
                bicycle_outputs = bicycle_model(torch_inputs).squeeze(0).numpy()[3:6]
            outputs -= bicycle_outputs

            self.learn_from_data(pos_info, inputs, outputs)

    def initialize(self, track, reference):
        settings = Settings(mpc_N=config.mpc_N, mpc_sample_time=config.mpc_sample_time,
                            bicycle_params=json.dumps(config.bicycle_params).encode('utf-8'),
                            n_midpoints=len(track), midpoints=track.astype(np.float32).tobytes(),
                            n_refpoints=len(reference), refpoints=reference.astype(np.float32).tobytes(),
                            use_gp=self.use_gp,
                            constraint_tightening=self.constraint_tightening)
        result = self.stub.initialize_solver(settings)
        if result.status == 0:
            raise Exception("Failed to make solver")

    def get_control(self, initial_state, max_speed=None):
        if max_speed is None:
            max_speed = config.speed_limit
        problem = Problem(
            initial_state=initial_state.astype(np.float32).tobytes(), max_speed=max_speed,
            delay_compensation=self.delay_compensation)
        solution = self.stub.solve(problem)
        state_horizon = np.frombuffer(solution.state_horizon, dtype=np.float32).reshape(config.mpc_N + 1, 9)
        control_horizon = np.frombuffer(solution.control_horizon, dtype=np.float32).reshape(config.mpc_N, 3)
        track_tighteners = np.frombuffer(solution.track_tighteners, dtype=np.float32).reshape(config.mpc_N + 1)
        done_cause = None
        if not solution.success:
            print("Fails", self.failed_solves)
            self.failed_solves += 1
            if self.failed_solves > 1:
                done_cause = "failed solves"
        else:
            self.failed_solves = 0

        return state_horizon, control_horizon, track_tighteners, done_cause

    def learn_from_data(self, pos_info, inputs, outputs):
        self.stub.learn_from_data(LearningData(
            pos_info=pos_info.astype(np.float32).tobytes(),
            inputs=inputs.astype(np.float32).tobytes(),
            outputs=outputs.astype(np.float32).tobytes())
        )

    def kill(self):
        print("DONE CALLED")
        self.stub.done(Empty())