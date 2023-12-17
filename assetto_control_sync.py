import numpy as np
import vgamepad as vg
import time


if __name__ == "__main__":
    output_controller = vg.VX360Gamepad()
    while True:
        output_controller.left_joystick_float(x_value_float=np.random.rand(), y_value_float=np.random.rand())
        output_controller.left_trigger_float(np.random.rand())
        output_controller.right_trigger_float(np.random.rand())
        time.sleep(0.5)