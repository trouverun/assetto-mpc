import time
import config
import numpy as np
import vgamepad as vg
from simulators.simulator import Simulator
from simulators.assetto_simulator.assettomemory import accSharedMemory



'''
Provides an interface to the "assetto corsa competizione.exe" running in the background by using the game shared memory.

Inputs are sent using an emulated game controller, which needs to already be synced with the game.

'''
class AssettoSimulator(Simulator):
    def __init__(self, shared_car_state, controls_queue, track, track_name, initial_max_speed):
        super().__init__(controls_queue, track, track_name, initial_max_speed)
        self.shared_car_state = shared_car_state
        self.asm = accSharedMemory()
        self.output_controller = vg.VX360Gamepad()
        self.output_controller.left_joystick_float(x_value_float=0, y_value_float=0)
        self.output_controller.left_trigger_float(0)
        self.output_controller.right_trigger_float(0)
        self.hdg = None
        self.steer = 0
        self.throttle = 0
        self.solve_time = 0
        self.current_gear = 0
        self.target_gear = 0
        self.last_gear_command = None
        self.last_gear_read = None
        self.iters_until_allowed_gear_change = 0
        self.prev_pos = None
        self.stuck_steps = 0

    def reset(self):
        super().reset()

    def read_state(self):
        timestamp = time.time_ns()
        sm = self.asm.read_shared_memory()
        if sm is None:
            raise ConnectionRefusedError("Shared memory not read")

        done_cause = None
        if self.done_event.is_set():
            done_cause = "requested by controller"

        if self.start_time is None:
            self.start_time = timestamp

        car_id = sm.Graphics.player_car_id
        idx = sm.Graphics.car_id.index(car_id)
        pos = sm.Graphics.car_coordinates[idx]

        track_pos = self.get_track_pos(np.array([pos.z, pos.x]))
        if len(self.lap_times) > config.num_eval_laps and self.max_speed == config.speed_limit:
            done_cause = "all laps done"

        accel_commands = [sm.Physics.gas, sm.Physics.brake]
        max_i = np.argmax(accel_commands)
        accel_command = accel_commands[max_i]
        if max_i == 1:
            accel_command *= -1

        if self.hdg is None:
            self.hdg = sm.Physics.heading
        else:
            self.hdg -= np.arctan2(np.sin(self.hdg - sm.Physics.heading), np.cos(self.hdg - sm.Physics.heading))

        pos_arr = np.array([pos.z, pos.x])
        if self.prev_pos is not None:
            if np.abs(self.prev_pos - pos_arr).sum() < 1e-3:
                self.stuck_steps += 1
                if self.stuck_steps > 100:
                    done_cause = "stuck"
            else:
                self.stuck_steps = 0
        self.prev_pos = pos_arr

        car_state = np.array([
            timestamp,
            pos.z,
            pos.x,
            -self.hdg,
            sm.Physics.local_velocity.z,
            sm.Physics.local_velocity.x,
            sm.Physics.local_angular_vel.y,
            9.8*sm.Physics.g_force.z,
            9.8*sm.Physics.g_force.x,
            0,
            -np.clip(sm.Physics.steer_angle, -1, 1),
            self.throttle,
            track_pos,
            self.max_speed
        ])

        if self.shared_car_state is not None:
            self.shared_car_state.set_state(car_state[:14])
        self._store_state(car_state[:14])

        # Gear change state machine
        if self.last_gear_read is None or (timestamp - self.last_gear_read) / 1e6 > config.gear_read_dt_ms:
            if sm.Physics.gear != 0:
                self.last_gear_read = timestamp
                current_gear = sm.Physics.gear
                if self.current_gear == 1:
                    self.target_gear = 2
                else:
                    if current_gear >= 2 and sm.Physics.rpm > config.gear_high_rpm:
                        self.target_gear = current_gear + 1
                        # print(self.current_gear, self.target_gear, sm.Physics.rpm)
                    elif current_gear > 2 and sm.Physics.rpm < config.gear_low_rpm:
                        self.target_gear = current_gear - 1
                        # print(self.current_gear, self.target_gear, sm.Physics.rpm)
                    else:
                        self.target_gear = current_gear
                self.current_gear = current_gear

        return car_state[:13], self.solve_time, done_cause

    def _set_controls(self, steer, throttle, d_steer, solve_time):
        self.steer = steer
        self.throttle = throttle
        self.output_controller.left_joystick_float(x_value_float=self.steer, y_value_float=0)
        if self.throttle >= 0:
            self.output_controller.left_trigger_float(0)
            self.output_controller.right_trigger_float(self.throttle)
        else:
            self.output_controller.right_trigger_float(0)
            self.output_controller.left_trigger_float(-self.throttle)

        if self.last_gear_command is not None:
            self.output_controller.release_button(self.last_gear_command)
            self.last_gear_command = None
            self.iters_until_allowed_gear_change = int((config.gear_change_dt_ms/1e3) / config.mpc_sample_time)
        else:
            if self.iters_until_allowed_gear_change == 0:
                if self.current_gear < self.target_gear:
                    # print("CHANGE UP!")
                    self.output_controller.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
                    self.last_gear_command = vg.XUSB_BUTTON.XUSB_GAMEPAD_A
                elif self.current_gear > self.target_gear:
                    # print("CHANGE DOWN")
                    self.output_controller.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
                    self.last_gear_command = vg.XUSB_BUTTON.XUSB_GAMEPAD_X
            else:
                self.iters_until_allowed_gear_change -= 1

        self.output_controller.update()
        self.solve_time = solve_time