import numpy as np
import config
import random
from utils.maths import zero_phase_filtering, save_spectrum_plot
from scipy.signal import savgol_filter
from abc import ABC, abstractmethod


def sm_array(shared_mem, shape, dtype=np.float32):
    return np.ndarray(shape, dtype=dtype, buffer=shared_mem.buf)


class CarStateSmWrapper(ABC):
    N_STATES = 14
    N_BYTES = N_STATES * 4
    def __init__(self, shared_mem, lock):
        self.sm = sm_array(shared_mem, 14)
        self.lock = lock

    def _index(self, i):
        with self.lock:
            val = self.sm[i]
        return val

    def set_state(self, state):
        with self.lock:
            self.sm[:] = state

    @property
    def controller_state(self):
        with self.lock:
            state = np.r_[self.sm[1:7].copy(), self.sm[10:13].copy()]

        # Model is ill defined at near zero velocities, simple fix:
        if state[3] < 5:
            state[3] += 5 - state[3]

        return state

    @property
    def max_speed(self):
        return self._index(13)


def generate_randomly_spaced_indices(M, N, chunks):
    total_occupied = chunks * N

    if total_occupied > M:
        raise ValueError("The specified number of chunks and size cannot fit within M indices.")

    remaining_space = M - total_occupied

    indices = []
    current_idx = 0

    for chunk in range(chunks):
        # Append N indices
        for i in range(current_idx, current_idx + N):
            indices.append(i)

        current_idx += N  # Move to the end of current chunk

        # If this isn't the last chunk, skip some indices
        if chunk < chunks - 1:
            # Randomly determine the skip value
            skip = random.randint(0, remaining_space // (chunks - chunk - 1))
            remaining_space -= skip
            current_idx += skip  # Move ahead by the skip value

    return indices


def filter_data_batch(batch, data_idx, low_pass_window_size, source_dir=None, data_names=None):
    tmp_original = None
    tmp_filtered = None
    dt_ns = (batch[1:, 0] - batch[:-1, 0])

    # print(np.mean(dt_ns) / 1e6, np.std(dt_ns) / 1e6)

    window_size = int(1e9*config.savgol_p / np.mean(dt_ns))

    for idx, deriv in data_idx:
        selected = batch[:, idx].copy()

        if source_dir is not None and data_names is not None:
            if idx in data_names.keys():
                save_spectrum_plot(selected, 1e9 / np.mean(dt_ns), source_dir + "/data_spectrum_%s.png" % data_names[idx])

        selected = zero_phase_filtering(selected, 2, 1e9 / np.mean(dt_ns), low_pass_window_size, 3)

        if deriv:
            savgol = 1e9*savgol_filter(
                selected, window_length=window_size, polyorder=config.savgol_d_k, deriv=1
            )[:-1] / dt_ns.mean()
            # Use finite difference derivative as reference:
            selected = 1e9 * (selected[1:] - selected[:-1]) / dt_ns.mean()
            selected = np.r_[selected, 0]
        else:
            # Filter all but hdg:
            savgol = savgol_filter(
                selected, window_length=window_size, polyorder=config.savgol_k, deriv=0
            )[:-1]

        if tmp_original is None:
            tmp_original = selected[:-1]
            tmp_filtered = savgol
        else:
            tmp_original = np.c_[tmp_original, selected[:-1]]
            tmp_filtered = np.c_[tmp_filtered, savgol]

    return tmp_original, tmp_filtered