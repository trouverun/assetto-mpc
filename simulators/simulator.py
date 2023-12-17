import numpy as np
import config
import time
import json
from datetime import datetime
import os
from simulators.pygame_renderer import Renderer
from threading import Thread, Event
from scipy import interpolate
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize



'''
Base class for the simulation handles. Provides a common interface to the ODE based simulation and ACC.

Implements a thread to manage the control queue and the pygame 2d visualization.

'''
class Simulator(ABC):
    def __init__(self, controls_queue, track, track_name, initial_max_speed):
        self.controls_queue = controls_queue
        self.recorded_states = [[]]
        self.solve_times = []
        self.lap_times = []
        self.start_time = None
        self.last_track_pos = None
        self.max_speed = initial_max_speed

        self.interpolated_track = None
        if track is not None:
            date_time = datetime.now().strftime("%d_%m_%Y_%H_%M")
            self.result_folder = "results/%s_%s" % (track_name, date_time)

            self.cx_spline = interpolate.splrep(track[:, 0], track[:, 1], k=3)
            self.cy_spline = interpolate.splrep(track[:, 0], track[:, 2], k=3)
            self.cw_spline = interpolate.splrep(track[:, 0], track[:, 3], k=3)
            positions = np.linspace(0, track[-1, 0], int(track[-1, 0]))
            x = interpolate.splev(positions, self.cx_spline, der=0)
            y = interpolate.splev(positions, self.cy_spline, der=0)
            dx = interpolate.splev(positions, self.cx_spline, der=1)
            dy = interpolate.splev(positions, self.cy_spline, der=1)
            hdg = np.arctan2(dy, dx)
            w = interpolate.splev(positions, self.cw_spline, der=0)
            self.interpolated_track = np.c_[positions, x, y, w, hdg]

        self.done_event = Event()
        self.control_thread = Thread(target=self.control_thread, args=(track,))
        self.control_thread.start()

    def control_thread(self, track):
        position_vel_history = []
        renderer = None
        if track is not None:
            renderer = Renderer()

        while True:
            controls = self.controls_queue.get()
            if controls is None:
                return

            state_horizon, control_horizon, track_tighteners, done_cause, solve_time, timestamp = controls

            time_now = time.time_ns()

            # print("delay: %d ms", ((time_now-timestamp)/1e6))

            if done_cause is not None:
                self.done_event.set()
            car_pos_hdg = state_horizon[0, :3]
            steer = state_horizon[1, 6]
            throttle = state_horizon[1, 7]
            d_steer = control_horizon[0, 0]
            # d_throttle = control_horizon[0, 1]
            self._set_controls(steer, throttle, d_steer, solve_time)
            self.solve_times.append(solve_time)

            if renderer is not None:
                position_vel_horizon = np.c_[state_horizon[:, :2], np.sqrt((state_horizon[:, 3] + state_horizon[:, 4])**2)]
                position_vel_history.append(position_vel_horizon[0].copy())

                tightened_x = interpolate.splev(state_horizon[:, -1], self.cx_spline)
                tightened_y = interpolate.splev(state_horizon[:, -1], self.cy_spline)
                tightened_w = interpolate.splev(state_horizon[:, -1], self.cw_spline) - track_tighteners
                tightened_dx = interpolate.splev(state_horizon[:, -1], self.cx_spline, der=1)
                tightened_dy = interpolate.splev(state_horizon[:, -1], self.cy_spline, der=1)
                tightened_hdg = np.arctan2(tightened_dy, tightened_dx)
                m1x = tightened_x + np.cos(tightened_hdg + np.pi / 2) * tightened_w
                m1y = tightened_y + np.sin(tightened_hdg + np.pi / 2) * tightened_w
                m2x = tightened_x + np.cos(tightened_hdg - np.pi / 2) * tightened_w
                m2y = tightened_y + np.sin(tightened_hdg - np.pi / 2) * tightened_w
                left = np.c_[m1x, m1y]
                right = np.c_[m2x, m2y]

                renderer.render_scenario(
                    car_pos_hdg, track, np.asarray(position_vel_history[-2*config.mpc_N:]), position_vel_horizon, left, right
                )

    def get_track_pos(self, car_pos, time_passed=None):
        if self.interpolated_track is None:
            return 0

        distances = np.sqrt(np.sum(np.square(self.interpolated_track[:, 1:3] - car_pos), axis=1))
        i = np.argmin(distances)
        track_pos = self.interpolated_track[i, 0]

        if self.last_track_pos is not None:
            if track_pos - self.last_track_pos < -self.interpolated_track[-1, 0] / 2:
                time_now = time.time_ns()
                if time_passed is None:
                    self.lap_times.append((time_now - self.start_time) / 1e9)
                else:
                    offset = 0
                    if len(self.lap_times) > 0:
                        offset = sum(self.lap_times)
                    self.lap_times.append(time_passed - offset)
                    print(f"LAPTIME {time_passed - offset} s")
                self.start_time = time_now
                self.recorded_states.append([])
                self.max_speed = min(self.max_speed + config.max_speed_increment, config.speed_limit)
        self.last_track_pos = track_pos

        return track_pos

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def read_state(self):
        pass

    @abstractmethod
    def _set_controls(self, steer, throttle, d_steer, solve_time):
        pass

    def _store_state(self, state):
        self.recorded_states[len(self.lap_times)].append(state)

    '''
    Makes fancy result plots
    '''
    def close(self, sim_type):
        if self.result_folder is not None:

            loop_times = self.lap_times
            if len(self.lap_times) <= config.num_eval_laps or self.max_speed != config.speed_limit:
                loop_times.append(0)

            result = {}

            for i, laptime in enumerate(loop_times):
                if laptime != 0:
                    result[f"lap_{i}"] = laptime

                lap_folder = f"{self.result_folder}/{sim_type}/lap_{i}"
                os.makedirs(lap_folder, exist_ok=True)

                plt.figure(figsize=(20, 10))
                plt.gcf().tight_layout()
                angle = np.deg2rad(-124.5)
                R = np.array([
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]
                ])
                cx = self.interpolated_track[:, 1]
                cy = self.interpolated_track[:, 2]
                cw = self.interpolated_track[:, 3]
                c_hdg = self.interpolated_track[:, 4]
                m1x = cx + np.cos(c_hdg + np.pi / 2) * cw
                m1y = cy + np.sin(c_hdg + np.pi / 2) * cw
                m2x = cx + np.cos(c_hdg - np.pi / 2) * cw
                m2y = cy + np.sin(c_hdg - np.pi / 2) * cw

                m1 = (R @ np.c_[m1x, m1y].T).T
                m2 = (R @ np.c_[m2x, m2y].T).T
                plt.scatter(m1[:, 0], m1[:, 1], c='black', s=10)
                plt.scatter(m2[:, 0], m2[:, 1], c='black', s=10)
                recorded_states = np.asarray(self.recorded_states[i])
                x = recorded_states[:, 1]
                y = recorded_states[:, 2]
                vx = recorded_states[:, 4]
                vy = recorded_states[:, 5]
                speed = np.sqrt(np.square(vx + vy))
                plt.title("Lap time %.2f seconds" % laptime)
                xy = (R @ np.c_[x, y].T).T
                cmap = plt.colormaps["cool"]
                plt.scatter(xy[:, 0], xy[:, 1], c=cmap(1 / config.speed_limit * speed), s=10)
                plt.colorbar(plt.cm.ScalarMappable(norm=Normalize(np.amin(speed) / config.KM_H, np.amax(speed) / config.KM_H), cmap=cmap), label="Car speed (Km/h)")
                plt.savefig(lap_folder + "/laps_plot.svg", bbox_inches='tight')
                plt.close()

                plt.figure(figsize=(20, 20))
                plt.gcf().tight_layout()
                ax = recorded_states[:, 7]
                ay = recorded_states[:, 8]
                valid = (ax > -30) & (ax < 30) & (ay > -30) & (ay < 30)
                ax = ax[valid]
                ay = ay[valid]

                g = 9.8  # gravitational constant in m/s^2
                # Convert to units of g
                ax_g = ax / g
                ay_g = ay / g
                plt.scatter(ay_g, ax_g)
                # Set axis limits
                plt.xlim(-2.5, 2.5)
                plt.ylim(-1.5, 1.5)
                # Add axis labels
                plt.xlabel("Lateral acceleration (g)")
                plt.ylabel("Longitudinal acceleration(g)")

                plt.savefig(lap_folder + "/gg_plot.svg", bbox_inches='tight')
                plt.close()

                np.save(f"{lap_folder}/states.npy", recorded_states)

            best4 = np.asarray(sorted(self.lap_times[1:])[:9])
            result["mean_lap_time_best9"] = best4.mean()
            result["std_lap_time_best9"] = best4.std()
            solve_array = np.asarray(self.solve_times)
            result["mean_solve_time"] = solve_array.mean()
            result["std_solve_time"] = solve_array.std()

            with open(f"{self.result_folder}/{sim_type}/result.json", "w") as f:
                f.write(json.dumps(result, indent=4))

        self.controls_queue.put(None)
        self.control_thread.join()