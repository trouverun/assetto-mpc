from inputs import get_gamepad
import math
import threading
import numpy as np


class XboxController(object):
    MAX_TRIG_VAL = math.pow(2, 8)
    MAX_JOY_VAL = math.pow(2, 15)

    def __init__(self):
        self.LeftJoystickX = 0
        self.LeftTrigger = 0
        self.RightTrigger = 0
        self.B = 0

        self._monitor_thread = threading.Thread(target=self._monitor_controller, args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def read(self):
        if self.RightTrigger >= self.LeftTrigger:
            throttle_value = np.power(self.RightTrigger, 2)
        else:
            throttle_value = -np.power(self.LeftTrigger, 2)

        if self.LeftJoystickX >= 0:
            steer_value = np.power(self.LeftJoystickX, 2)
        else:
            steer_value = -np.power(-self.LeftJoystickX, 2)

        return [steer_value,
                throttle_value,
                self.B]

    def _monitor_controller(self):
        while True:
            events = get_gamepad()
            for event in events:
                if event.code == 'ABS_X':
                    self.LeftJoystickX = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_Z':
                    self.LeftTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1
                elif event.code == 'ABS_RZ':
                    self.RightTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1
                elif event.code == 'BTN_EAST':
                    self.B = event.state
