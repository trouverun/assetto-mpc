import config
import numpy as np
from utils.general import filter_data_batch


class LearningStorage:
    def __init__(self,):
        self.current_data = np.zeros([config.learning_low_pass_window, 13])
        self.n_data = 0

    def get_real_time_learning_sample(self, data):
        data_idx = [
            # info:
            (12, False),
            (1, False),
            (2, False),
            # Inputs (0-5):
            (4, False),   # VX
            (5, False),   # VY
            (6, False),   # W
            (10, False),  # STEER
            (11, False),  # THROTTLE
            # Outputs (5-8):
            (7, False),   # AX
            (8, False),   # AY
            (6, True),    # dW
        ]

        if self.n_data < config.learning_low_pass_window:
            self.current_data[self.n_data] = data
            self.n_data += 1
            return None, None, None

        self.current_data[:-1] = self.current_data[1:]
        self.current_data[-1] = data

        _, tmp_filtered = filter_data_batch(self.current_data, data_idx, config.learning_low_pass_window)
        filtered = tmp_filtered[len(tmp_filtered) // 2]

        return filtered[:3], filtered[3:8], filtered[8:]


class MockLearningStorage:
    def __init__(self):
        pass

    def get_real_time_learning_sample(self, data):
        return None, None