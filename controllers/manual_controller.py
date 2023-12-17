import numpy as np
import config
from controllers.controller import Controller
from utils.game_controller import XboxController


class ManualController(Controller):
    def __init__(self):
        self.input_controller = XboxController()
        self.steer, self.throttle = 0, 0

    def initialize(self, track, reference):
        pass

    def get_control(self, initial_state, max_speed=None):
        new_steer, new_throttle, done = self.input_controller.read()

        if np.abs(new_steer - self.steer) > config.u_steer_max * config.mpc_sample_time:
            d_steer = np.sign(new_steer - self.steer) * config.u_steer_max * config.mpc_sample_time
            self.steer += d_steer
        else:
            d_steer = new_steer - self.steer
            self.steer += d_steer

        if np.abs(new_throttle - self.throttle) > config.u_throttle_max * config.mpc_sample_time:
            d_throttle = np.sign(new_throttle - self.throttle) * config.u_throttle_max * config.mpc_sample_time
            self.throttle += d_throttle
        else:
            d_throttle = new_throttle - self.throttle
            self.throttle += d_throttle

        self.steer = np.clip(self.steer, -config.steer_max, config.steer_max)
        self.throttle = np.clip(self.throttle, -config.throttle_max, config.throttle_max)

        done_cause = None
        if done:
            done_cause = "requested by user"

        state_horizon = np.zeros([config.mpc_N + 1, 9])
        state_horizon[:, :5] = initial_state[:5].copy()
        state_horizon[1:, 6] = self.steer
        state_horizon[1:, 7] = self.throttle

        control_horizon = np.zeros([config.mpc_N, 3])
        control_horizon[0, 0] = d_steer
        control_horizon[0, 1] = d_throttle

        track_tighteners = np.zeros(config.mpc_N + 1)

        return state_horizon, control_horizon, track_tighteners, done_cause

    def learn_from_data(self, inputs, outputs):
        pass
