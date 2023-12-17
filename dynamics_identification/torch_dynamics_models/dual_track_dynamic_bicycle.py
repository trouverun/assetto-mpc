import config
import torch
import numpy as np
from utils.general import filter_data_batch


class Tire(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.B_x = torch.nn.Parameter(torch.tensor([config.bicycle_params["B_x"]]))
        self.C_x = torch.nn.Parameter(torch.tensor([config.bicycle_params["C_x"]]))
        self.B_y = torch.nn.Parameter(torch.tensor([config.bicycle_params["B_y"]]))
        self.C_y = torch.nn.Parameter(torch.tensor([config.bicycle_params["C_y"]]))
        self.Fzf0 = torch.nn.Parameter(torch.tensor([config.bicycle_params["Fzf0"]]))

    def forward(self, tensor):
        Fz = tensor[:, 0]
        slip = tensor[:, 1]
        slip_angle = tensor[:, 2]
        mu = tensor[:, 3]

        # Friction force does not scale linearly with normal force since the tyre will deform at high normal forces:
        Fz_adjusted = Fz * (1 - self.eps * (Fz / self.Fzf0))

        Fx = Fz_adjusted * mu * torch.sin(self.C_y * torch.arctan(self.B_y * slip))
        Fy = Fz_adjusted * mu * torch.sin(self.C_y * torch.arctan(self.B_y * slip_angle))

        return Fx, Fy


class Drivetrain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.torque_a = torch.nn.Parameter(torch.tensor([config.bicycle_params["torque_a"]]))
        self.torque_b = torch.nn.Parameter(torch.tensor([config.bicycle_params["torque_b"]]))
        self.torque_c = torch.nn.Parameter(torch.tensor([config.bicycle_params["torque_c"]]))
        self.torque_d = torch.nn.Parameter(torch.tensor([config.bicycle_params["torque_d"]]))
        self.torque_e = torch.nn.Parameter(torch.tensor([config.bicycle_params["torque_e"]]))
        self.rpm_a = torch.nn.Parameter(torch.tensor([config.bicycle_params["rpm_a"]]))
        self.rpm_b = torch.nn.Parameter(torch.tensor([config.bicycle_params["rpm_b"]]))

    def forward(self, tensor):
        throttle = tensor[:, 0]
        rpm = tensor[:, 1]

        d_rpm = self.rpm_a * throttle - self.rpm_b * rpm

        torque = throttle**self.torque_e * (
            self.a / (1 + torch.exp(-(rpm - self.b) / self.c) * -self.d * rpm)
        )

        return d_rpm, torque


class DualTrackBicycle(torch.nn.Module):
    def __init__(self, sim_mode=False):
        super().__init__()
        self.sim_mode = sim_mode
        self.car_lf = torch.nn.Parameter(torch.tensor([config.bicycle_params["car_lf"]]))
        self.car_lr = torch.nn.Parameter(torch.tensor([config.bicycle_params["car_lr"]]))
        self.track_width = torch.nn.Parameter(torch.tensor([config.bicycle_params["track_width"]]))
        self.car_max_steer = torch.nn.Parameter(torch.tensor([config.bicycle_params["car_max_steer"]]))
        self.car_mass = torch.nn.Parameter(torch.tensor([config.bicycle_params["car_mass"]]))
        self.car_inertia = torch.nn.Parameter(torch.tensor([config.bicycle_params["car_inertia"]]))

        # Left and right tires share params:
        self.f_tire = Tire()
        self.r_tire = Tire()

        self.drivetrain = Drivetrain()

        self.d_steer_k_0 = torch.nn.Parameter(torch.tensor([config.bicycle_params["d_steer_k_0"]]))
        self.d_steer_k_1 = torch.nn.Parameter(torch.tensor([config.bicycle_params["d_steer_k_1"]]))
        self.yaw_saturation_k = torch.nn.Parameter(torch.tensor([config.bicycle_params["yaw_saturation_k"]]))
        self.hf = torch.nn.Parameter(torch.tensor([config.bicycle_params["hf"]]))
        self.hr = torch.nn.Parameter(torch.tensor([config.bicycle_params["hr"]]))
        self.muf = torch.nn.Parameter(torch.tensor([config.bicycle_params["muf"]]))
        self.mur = torch.nn.Parameter(torch.tensor([config.bicycle_params["mur"]]))
        self.eps = torch.nn.Parameter(torch.tensor([config.bicycle_params["eps"]]))
        self.Fzf0 = torch.nn.Parameter(torch.tensor([config.bicycle_params["Fzf0"]]))
        self.Fzr0 = torch.nn.Parameter(torch.tensor([config.bicycle_params["Fzr0"]]))

        self.wheel_r = torch.nn.Parameter(torch.tensor([config.bicycle_params["wheel_r"]]))
        self.wheel_inertia = torch.nn.Parameter(torch.tensor([config.bicycle_params["wheel_inertia"]]))

        self.front_wheel_hub_friction = torch.nn.Parameter(torch.tensor([config.bicycle_params["front_wheel_hub_friction"]]))
        self.rear_wheel_hub_friction = torch.nn.Parameter(torch.tensor([config.bicycle_params["rear_wheel_hub_friction"]]))

        self.fixed_gear_ratio = torch.nn.Parameter(torch.tensor([config.bicycle_params["fixed_gear_ratio"]]))
        self.fixed_gear_efficiency = torch.nn.Parameter(torch.tensor([config.bicycle_params["fixed_gear_efficiency"]]))

        self.drag_coefficient = torch.nn.Parameter(torch.tensor([config.bicycle_params["drag_coefficient"]]))

        self.d_steer = 0

    def setup_params(self, x):
        self.car_mass = torch.nn.Parameter(x[0])
        self.car_max_steer = torch.nn.Parameter(x[1])
        self.car_lf = torch.nn.Parameter(x[2])
        self.car_lr = torch.nn.Parameter(x[3])
        self.track_width = torch.nn.Parameter(x[4])
        self.car_inertia = torch.nn.Parameter(x[5])
        self.wheel_Bf = torch.nn.Parameter(x[6])
        self.wheel_Cf = torch.nn.Parameter(x[7])
        self.wheel_Br = torch.nn.Parameter(x[8])
        self.wheel_Cr = torch.nn.Parameter(x[9])
        self.d_steer_k_0 = torch.nn.Parameter(x[10])
        self.d_steer_k_1 = torch.nn.Parameter(x[11])
        self.yaw_saturation_k = torch.nn.Parameter(x[12])
        self.hf = torch.nn.Parameter(x[13])
        self.hr = torch.nn.Parameter(x[14])
        self.muf = torch.nn.Parameter(x[15])
        self.mur = torch.nn.Parameter(x[16])
        self.eps = torch.nn.Parameter(x[17])
        self.Fzf0 = torch.nn.Parameter(x[18])
        self.Fzr0 = torch.nn.Parameter(x[19])
        self.wheel_r = torch.nn.Parameter(x[20])
        self.wheel_inertia = torch.nn.Parameter(x[21])
        self.front_wheel_hub_friction = torch.nn.Parameter(x[22])
        self.rear_wheel_hub_friction = torch.nn.Parameter(x[23]),
        self.fixed_gear_ratio = torch.nn.Parameter(x[24]),
        self.fixed_gear_efficiency = torch.nn.Parameter(x[25])
        self.drag_coefficient = torch.nn.Parameter(x[26])

    def extract_params(self):
        return torch.tensor([
            self.car_mass.cpu().item(),
            self.car_max_steer.cpu().item(),
            self.car_lf.cpu().item(),
            self.car_lr.cpu().item(),
            self.track_width.cpu().item(),
            self.car_inertia.cpu().item(),
            self.wheel_Bf.cpu().item(),
            self.wheel_Cf.cpu().item(),
            self.wheel_Br.cpu().item(),
            self.wheel_Cr.cpu().item(),
            self.d_steer_k_0.cpu().item(),
            self.d_steer_k_1.cpu().item(),
            self.yaw_saturation_k.cpu().item(),
            self.hf.cpu().item(),
            self.hr.cpu().item(),
            self.muf.cpu().item(),
            self.mur.cpu().item(),
            self.eps.cpu().item(),
            self.Fzf0.cpu().item(),
            self.Fzr0.cpu().item(),
            self.wheel_r.cpu().item(),
            self.wheel_inertia.cpu().item(),
            self.front_wheel_hub_friction.cpu().item(),
            self.rear_wheel_hub_friction.cpu().item(),
            self.fixed_gear_ratio.cpu().item(),
            self.fixed_gear_efficiency.cpu().item(),
            self.drag_coefficient.cpu().item()
        ])

    def forward(self, tensor):
        hdg = tensor[:, 0]
        vx = tensor[:, 1]
        vy = -tensor[:, 2]
        w = tensor[:, 3]
        steer = tensor[:, 4]
        throttle = tensor[:, 5]
        brake = tensor[:, 6]
        d_steer = tensor[:, 7]
        roll = tensor[:, 8]
        pitch = tensor[:, 9]
        rpm = tensor[:, 10]
        omega_fl = tensor[:, 11]
        omega_fr = tensor[:, 12]
        omega_rl = tensor[:, 13]
        omega_rr = tensor[:, 14]

        g = 9.8
        true_steer = steer * self.car_max_steer

        # Wheel slips:
        v_f = torch.cos(true_steer) * vx + torch.sin(true_steer) * vy
        v_r = vx
        s_fl = (omega_fl * self.wheel_r - v_f) / torch.max(v_f, omega_fl * self.wheel_r)
        s_fr = (omega_fr * self.wheel_r - v_f) / torch.max(v_f, omega_fr * self.wheel_r)
        s_rl = (omega_rl * self.wheel_r - v_r) / torch.max(v_r, omega_rl * self.wheel_r)
        s_rr = (omega_rr * self.wheel_r - v_r) / torch.max(v_r, omega_rr * self.wheel_r)
        # Wheel slip angles:
        vx_frontwheel = vx*torch.cos(true_steer) + (vy + w*self.car_lf)*torch.sin(true_steer)
        vy_frontwheel = (vy + w*self.car_lf)*torch.cos(true_steer) - vx*torch.sin(true_steer)
        a_f = torch.arctan2(vy_frontwheel, vx_frontwheel + 1e-5)
        a_r = torch.arctan2(vy - w * self.car_lr, vx + 1e-5)

        # Wheelbase:
        wheelbase = (self.car_lf + self.car_lr)
        W = self.car_mass * g
        # Load on front axle:
        Wf = W * self.car_lr / wheelbase
        # Load on rear axle:
        Wr = W * self.car_lf / wheelbase

        # Setting initial force guess as all 0 = no weight transfer
        Fy_fl, Fy_fr, Fy_rl, Fy_rr = torch.zeros_like(a_f), torch.zeros_like(a_f), torch.zeros_like(a_f), torch.zeros_like(a_f)
        Fx_fl, Fx_fr, Fx_rl, Fx_rr = torch.zeros_like(a_f), torch.zeros_like(a_f), torch.zeros_like(a_f), torch.zeros_like(a_f)
        for _ in range(5):
            # Common loads on front tires: nominal load + torque from road inclination + torque from longitudinal acceleration
            Fz_front = -W*torch.sin(pitch)/wheelbase*(self.hf+self.hr)/2 - self.hr*(Fx_rl+Fx_rr)/self.car_lf - self.hf*(Fx_fl+Fx_fr)/self.car_lr
            # Common loads on rear tires: nominal load + torque from road inclination + torque from longitudinal acceleration
            Fz_rear = W*torch.sin(pitch)/wheelbase*(self.hf+self.hr)/2 + self.hr*(Fx_rl+Fx_rr)/self.car_lf + self.hf*(Fx_fl+Fx_fr)/self.car_lr

            # Common loads on left tires: nominal load + torque from road inclination + torque from longitudinal acceleration
            Fz_left = W*torch.sin(-roll)/self.track_width*(self.hf+self.hr)/2 - self.hr*(Fy_rl+Fy_rr)/(self.track_width/2) - self.hf*(Fy_fl+Fy_fr)*torch.cos(steer*self.car_max_steer)/(self.track_width/2)
            # Common loads on right tires: nominal load + torque from road inclination + torque from longitudinal acceleration
            Fz_right = W*torch.sin(-roll)/self.track_width*(self.hf+self.hr)/2 + self.hr*(Fy_rl+Fy_rr)/(self.track_width/2) + self.hf*(Fy_fl+Fy_fr)*torch.cos(steer*self.car_max_steer)/(self.track_width/2)

            # Load on front left tire:
            Fz_fl = Wf/2 + Fz_front/2 + Fz_left/2
            # Load on front right tire
            Fz_fr = Wf/2 + Fz_front/2 + Fz_right/2
            # Load on rear left tire:
            Fz_rl = Wr/2 + Fz_rear/2 + Fz_left/2
            # Load on rear right tire:
            Fz_rr = Wr/2 + Fz_rear/2 + Fz_right/2

            fl_tensor = torch.vstack([Fz_fl, s_fl, a_f, self.muf]).T
            Fx_fl, Fy_fl = self.f_tire(fl_tensor)
            fr_tensor = torch.vstack([Fz_fr, s_fr, a_f, self.muf]).T
            Fx_fr, Fy_fr = self.f_tire(fr_tensor)
            rl_tensor = torch.vstack([Fz_rl, s_rl, a_r, self.muf]).T
            Fx_rl, Fy_rl = self.r_tire(rl_tensor)
            rr_tensor = torch.vstack([Fz_rr, s_rr, a_r, self.muf]).T
            Fx_rr, Fy_rr = self.r_tire(rr_tensor)

        drivetrain_tensor = torch.vstack([throttle, rpm]).T
        d_rpm, torque = self.drivetrain(drivetrain_tensor)

        drivetrain_torque = torque * self.fixed_gear_ratio * self.fixed_gear_efficiency
        front_brake_torque = self.front_brake_k1 * brake
        rear_brake_torque = self.rear_brake_k1 * brake

        d_omega_fl = (-Fx_fl * self.wheel_r - front_brake_torque - self.front_wheel_hub_friction * omega_fl) / self.wheel_inertia
        d_omega_fr = (-Fx_fr * self.wheel_r - front_brake_torque - self.front_wheel_hub_friction * omega_fr) / self.wheel_inertia
        d_omega_rl = ((drivetrain_torque / 2) - Fx_rl * self.wheel_r - rear_brake_torque - self.rear_wheel_hub_friction * omega_rl) / self.wheel_inertia
        d_omega_rr = ((drivetrain_torque / 2) - Fx_rr * self.wheel_r - rear_brake_torque - self.rear_wheel_hub_friction * omega_rr) / self.wheel_inertia

        Fx_drag = self.drag_coefficient * vx**2
        Fx_lnet = Fx_fl + Fx_rl
        Fx_rnet = Fx_fr + Fx_rr
        Fx_net = Fx_lnet + Fx_rnet + Fx_drag

        Fy_fnet = Fy_fl + Fy_fr
        Fy_rnet = Fy_rl + Fy_rr
        Fy_net = Fy_fnet + Fy_rnet

        output = torch.vstack([
            vx * torch.cos(hdg) - vy * torch.sin(hdg),
            vx * torch.sin(hdg) + vy * torch.cos(hdg),
            w,
            1 / self.car_mass * (Fx_net - Fy_net * torch.sin(true_steer)) - pitch * g,
            1 / self.car_mass * (Fy_rnet + Fy_fnet * torch.cos(true_steer)) + roll * g,
            1 / self.car_inertia * (Fy_fnet * self.car_lf * torch.cos(true_steer) - Fy_rnet * self.car_lr),
            d_rpm,
            d_omega_fl,
            d_omega_fr,
            d_omega_rl,
            d_omega_rr,
        ]).T

        if self.sim_mode:
            output[:, 3] += vy * w
            output[:, 4] -= vx * w
            output = torch.hstack([output, torch.zeros([len(output), 2]).to(output.device)])
        else:
            slips = torch.vstack([a_f, a_r]).T
            output = torch.hstack([output, slips])

        return output

    def extract_inputs_outputs(self, data, source_dirs):
        original_inputs = []
        original_outputs = []
        filtered_inputs = []
        filtered_outputs = []

        for batch, source in zip(data, source_dirs):
            data_idx = [
                # Inputs (0-6):
                (1, False),   # hdg
                (2, False),   # VX
                (3, False),   # VY
                (4, False),   # W
                (5, False),   # STEER
                (6, False),   # THROTTLE
                (7, False),   # Brake
                (5, True),    # D steer
                (8, False),   # r
                (9, False),   # p
                # Outputs (6-13):
            ]
            normal_batch = np.c_[
                batch['Timestamp'],
                batch['Heading'],  # hdg
                batch['Local_Velocity_Z'],  # VX
                batch['Local_Velocity_X'],  # VY
                batch['Local_Angular_Velocity_Y'],  # W
                batch['Steer_Angle'],  # STEER
                batch['Gas'],  # THROTTLE
                batch['Brake'],  # BRAKE
                batch['Roll'],  # roll
                batch['Pitch'],  # pitch
                batch['RPM'],   # ENG RPM
                batch['Wheel_Angular_Speed_Front_Left'],  # FL RPM
                batch['Wheel_Angular_Speed_Front_Right'],  # FR RPM
                batch['Wheel_Angular_Speed_Rear_Left'],  # RL RPM
                batch['Wheel_Angular_Speed_Rear_Right'],  # RR RPM
                batch['Local_Acceleration_Z'],  # AX
                batch['Local_Acceleration_X'],  # AY
                batch['Slip_Angle_Front_Left'],  # Slip angle
                batch['Slip_Angle_Front_Right'],  # Slip angle
                batch['Slip_Angle_Rear_Left'],  # Slip angle
                batch['Slip_Angle_Rear_Right'],  # Slip angle
                batch['Wheel_Slip_Front_Left'],  # Slip
                batch['Wheel_Slip_Front_Right'],  # Slip
                batch['Wheel_Slip_Rear_Left'],  # Slip
                batch['Wheel_Slip_Rear_Right'],  # Slip
            ]

            data_names = {
                2: "VX",
                3: "VY",
                4: "W",
                9: "AX",
                10: "AY"
            }

            tmp_original, tmp_filtered = filter_data_batch(normal_batch, data_idx, config.low_pass_window, source, data_names)
            original_inputs.extend(tmp_original[:, :9])
            original_outputs.extend(tmp_original[:, 9:])
            filtered_inputs.extend(tmp_filtered[:, :9])
            filtered_outputs.extend(tmp_filtered[:, 9:])

        original_inputs, original_outputs = np.asarray(original_inputs), np.asarray(original_outputs)
        filtered_inputs, filtered_outputs = np.asarray(filtered_inputs), np.asarray(filtered_outputs)
        original_outputs = np.c_[
            np.zeros([len(original_outputs), 3]), original_outputs[:, :-4],
            np.mean(original_outputs[:, -4:-2], axis=1), np.mean(original_outputs[:, -2:], axis=1)
        ]
        filtered_outputs = np.c_[
            np.zeros([len(filtered_outputs), 3]), filtered_outputs[:, :-4],
            np.mean(filtered_outputs[:, -4:-2], axis=1), np.mean(filtered_outputs[:, -2:], axis=1)
        ]

        return original_inputs, original_outputs, filtered_inputs, filtered_outputs






