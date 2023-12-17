import config
import torch
import numpy as np
from utils.general import filter_data_batch
from dynamics_identification.torch_dynamics_models.bicycle_model import BicycleModel


class DriveTrain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.car_Tm0 = torch.nn.Parameter(torch.tensor([config.bicycle_params["drivetrain.car_Tm0"]]))
        self.car_Tm1 = torch.nn.Parameter(torch.tensor([config.bicycle_params["drivetrain.car_Tm1"]]))
        self.car_Tr0 = torch.nn.Parameter(torch.tensor([config.bicycle_params["drivetrain.car_Tr0"]]))
        self.car_Tr1 = torch.nn.Parameter(torch.tensor([config.bicycle_params["drivetrain.car_Tr1"]]))
        self.car_Tr2 = torch.nn.Parameter(torch.tensor([config.bicycle_params["drivetrain.car_Tr2"]]))

    def setup_params(self, x):
        self.car_Tm0 = torch.nn.Parameter(x[0])
        self.car_Tm1 = torch.nn.Parameter(x[1])
        self.car_Tr0 = torch.nn.Parameter(x[2])
        self.car_Tr1 = torch.nn.Parameter(x[3])
        self.car_Tr2 = torch.nn.Parameter(x[4])

    def get_constraint_costs(self):
        cost = 0

        cost += 1e-5 * torch.abs(self.car_Tm0).item()
        cost += 1e-5 * torch.abs(self.car_Tm1).item()
        cost += 1e-5 * torch.abs(self.car_Tr0).item()
        cost += 1e-5 * torch.abs(self.car_Tr1).item()
        cost += 1e-5 * torch.abs(self.car_Tr2).item()

        return cost

    def forward(self, tensor):
        vx = tensor[:, 0] #/ 100
        throttle = tensor[:, 1]
        Fx = (
            ((self.car_Tm0 + self.car_Tm1 * vx) * throttle)
            - (self.car_Tr0*(1-torch.tanh(self.car_Tr1*vx)) + self.car_Tr2 * vx**2)
        )
        return torch.vstack([Fx]).T


class SingleTrackBicycle(BicycleModel):
    def __init__(self, sim_mode=False):
        super().__init__()
        self.sim_mode = sim_mode
        self.car_mass = torch.nn.Parameter(torch.tensor([config.bicycle_params["car_mass"]]))
        self.car_max_steer = torch.nn.Parameter(torch.tensor([config.bicycle_params["car_max_steer"]]))
        self.car_lf = torch.nn.Parameter(torch.tensor([config.bicycle_params["car_lf"]]))
        self.car_lr = torch.nn.Parameter(torch.tensor([config.bicycle_params["car_lr"]]))
        self.car_inertia = torch.nn.Parameter(torch.tensor([config.bicycle_params["car_inertia"]]))
        self.wheel_Bf = torch.nn.Parameter(torch.tensor([config.bicycle_params["wheel_Bf"]]))
        self.wheel_Cf = torch.nn.Parameter(torch.tensor([config.bicycle_params["wheel_Cf"]]))
        self.wheel_Df = torch.nn.Parameter(torch.tensor([config.bicycle_params["wheel_Df"]]))
        self.wheel_Br = torch.nn.Parameter(torch.tensor([config.bicycle_params["wheel_Br"]]))
        self.wheel_Cr = torch.nn.Parameter(torch.tensor([config.bicycle_params["wheel_Cr"]]))
        self.wheel_Dr = torch.nn.Parameter(torch.tensor([config.bicycle_params["wheel_Dr"]]))
        self.drivetrain = DriveTrain()

        self.d_steer = 0

    def setup_params(self, x):
        self.drivetrain.setup_params(x[:5])
        self.car_mass = torch.nn.Parameter(x[5])
        self.car_max_steer = torch.nn.Parameter(x[6])
        self.car_lf = torch.nn.Parameter(x[7])
        self.car_lr = torch.nn.Parameter(x[8])
        self.car_inertia = torch.nn.Parameter(x[9])
        self.wheel_Bf = torch.nn.Parameter(x[10])
        self.wheel_Cf = torch.nn.Parameter(x[11])
        self.wheel_Df = torch.nn.Parameter(x[12])
        self.wheel_Br = torch.nn.Parameter(x[13])
        self.wheel_Cr = torch.nn.Parameter(x[14])
        self.wheel_Dr = torch.nn.Parameter(x[15])

    def extract_params(self):
        return torch.tensor([
            self.drivetrain.car_Tm0.cpu().item(),
            self.drivetrain.car_Tm1.cpu().item(),
            self.drivetrain.car_Tr0.cpu().item(),
            self.drivetrain.car_Tr1.cpu().item(),
            self.drivetrain.car_Tr2.cpu().item(),
            self.car_mass.cpu().item(),
            self.car_max_steer.cpu().item(),
            self.car_lf.cpu().item(),
            self.car_lr.cpu().item(),
            self.car_inertia.cpu().item(),
            self.wheel_Bf.cpu().item(),
            self.wheel_Cf.cpu().item(),
            self.wheel_Df.cpu().item(),
            self.wheel_Br.cpu().item(),
            self.wheel_Cr.cpu().item(),
            self.wheel_Dr.cpu().item(),
        ])

    def get_variable_scalers(self):
        return torch.tensor([
            1e4,    # drivetrain.car_Tm0
            1e1,    # drivetrain.car_Tm1
            1e4,    # drivetrain.car_Tr0
            1e1,    # drivetrain.car_Tr1
            1e3,    # drivetrain.car_Tr2
            1e3,    # car_mass
            1,      # car_max_steer
            1.5,    # car_lf
            1.5,    # car_lr
            1.5e3,  # car_inertia
            1e1,    # wheel_Bf
            1,      # wheel_Cf
            5e3,    # wheel_Df
            1e1,    # wheel_Br
            1,      # wheel_Cr
            5e3,    # wheel_Dr
        ])

    def get_output_weights(self):
        return torch.tensor([0, 0, 0, 1, 1, 180/np.pi/2, 0, 0])

    def get_constraint_costs(self):
        cost = 0

        cost -= 100 * torch.min(torch.zeros(1).cuda(), (self.car_lf + self.car_lr) - 2.4).item()
        cost += 100 * torch.max(torch.zeros(1).cuda(), (self.car_lf + self.car_lr) - 2.8).item()
        cost -= 1000 * torch.min(torch.zeros(1).cuda(), self.car_lf - 0.75).item()
        cost -= 1000 * torch.min(torch.zeros(1).cuda(), self.car_lr - 0.75).item()
        cost += 1000 * torch.max(torch.zeros(1).cuda(), self.car_lr - self.car_lf).item()

        cost -= 100 * torch.min(torch.zeros(1).cuda(), torch.abs(self.car_max_steer) - 0.25).item()
        cost += 100 * torch.max(torch.zeros(1).cuda(), torch.abs(self.car_max_steer) - 1).item()

        cost -= 10 * torch.min(torch.zeros(1).cuda(), self.car_mass - 1000).item()
        cost += 10 * torch.max(torch.zeros(1).cuda(), self.car_mass - 1500).item()

        cost -= 10 * torch.min(torch.zeros(1).cuda(), self.car_inertia - 500).item()
        cost += 10 * torch.max(torch.zeros(1).cuda(), self.car_inertia - 3000).item()

        cost += 100 * torch.max(torch.zeros(1).cuda(), torch.abs(self.wheel_Cf) - 3).item()
        cost += 100 * torch.max(torch.zeros(1).cuda(), torch.abs(self.wheel_Cr) - 3).item()
        cost += 100 * torch.max(torch.zeros(1).cuda(), torch.abs(self.wheel_Bf) - 30).item()
        cost += 100 * torch.max(torch.zeros(1).cuda(), torch.abs(self.wheel_Br) - 30).item()
        cost -= 10 * torch.min(torch.zeros(1).cuda(), torch.abs(self.wheel_Df) - 2500).item()
        cost -= 10 * torch.min(torch.zeros(1).cuda(), torch.abs(self.wheel_Dr) - 2500).item()
        cost += 10 * torch.max(torch.zeros(1).cuda(), torch.abs(self.wheel_Df) - 10000).item()
        cost += 10 * torch.max(torch.zeros(1).cuda(), torch.abs(self.wheel_Dr) - 10000).item()

        cost += self.drivetrain.get_constraint_costs()

        return cost

    def forward(self, tensor):
        # print(tensor)
        hdg = tensor[:, 0]
        vx = tensor[:, 1]
        vy = tensor[:, 2]
        w = tensor[:, 3]
        steer = tensor[:, 4]
        throttle = tensor[:, 5]
        if not self.sim_mode:
            roll = tensor[:, 7]
            pitch = tensor[:, 8]

        true_steer = steer * self.car_max_steer

        drivetrain_tensor = torch.vstack([vx, throttle]).T
        Frx = self.drivetrain(drivetrain_tensor)[:, 0]

        vx_frontwheel = vx*torch.cos(true_steer) + (vy + w*self.car_lf)*torch.sin(true_steer)
        vy_frontwheel = (vy + w*self.car_lf)*torch.cos(true_steer) - vx*torch.sin(true_steer)
        af = torch.arctan2(vy_frontwheel, vx_frontwheel+2.5)
        af = torch.arctan2(vy + w*self.car_lf, vx+2.5) - steer*self.car_max_steer
        Ffy = self.wheel_Df * torch.sin(self.wheel_Cf * torch.arctan(self.wheel_Bf * af))

        ar = torch.arctan2(vy - w*self.car_lr, vx+2.5)
        Fry = self.wheel_Dr * torch.sin(self.wheel_Cr * torch.arctan(self.wheel_Br * ar))

        # print(af, ar, Ffy, Fry, vx, vy, w)

        output = torch.vstack([
             vx * torch.cos(hdg) - vy * torch.sin(hdg),
             vx * torch.sin(hdg) + vy * torch.cos(hdg),
             w,
             1 / self.car_mass * (Frx - Ffy * torch.sin(true_steer)),
             1 / self.car_mass * (Fry + Ffy * torch.cos(true_steer)),
             1 / self.car_inertia * (Ffy * self.car_lf * torch.cos(true_steer) - Fry * self.car_lr),
        ]).T


        if self.sim_mode:
            output[:, 3] += vy * w
            output[:, 4] -= vx * w
            output = torch.hstack([output, torch.zeros([len(output), 2]).to(output.device)])
        else:
            output[:, 3] -= torch.sin(pitch) * 9.8
            output[:, 4] += torch.sin(roll) * 9.8
            slips = torch.vstack([af, ar]).T
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
                (5, True),    # D steer
                (7, False),   # r
                (8, False),   # p
                # Outputs (6-13):
                (9, False),   # AX
                (10, False),  # AY
                (4, True),    # dW
                (11, False),  # Slip
                (12, False),  # Slip
                (13, False),  # Slip
                (14, False),  # Slip
            ]
            source_batch = np.c_[
                batch['Timestamp'],
                batch['Heading'],                          # hdg
                batch['Local_Velocity_Z'],                 # VX
                batch['Local_Velocity_X'],                 # VY
                batch['Local_Angular_Velocity_Y'],         # W
                batch['Steer_Angle'],                      # STEER
                batch['Gas'] - batch['Brake'],             # THROTTLE
                batch['Roll'],                             # roll
                batch['Pitch'],                            # pitch
                batch['Local_Acceleration_Z'],             # AX
                batch['Local_Acceleration_X'],             # AY
                batch['Slip_Angle_Front_Left'],            # Slip
                batch['Slip_Angle_Front_Right'],           # Slip
                batch['Slip_Angle_Rear_Left'],             # Slip
                batch['Slip_Angle_Rear_Right'],            # Slip
            ]

            data_names = {
                2: "VX",
                3: "VY",
                4: "W",
                9: "AX",
                10: "AY"
            }

            tmp_original, tmp_filtered = filter_data_batch(source_batch, data_idx, config.low_pass_window, source, data_names)
            original_inputs.extend(tmp_original[:, :9])
            original_outputs.extend(tmp_original[:, 9:])
            filtered_inputs.extend(tmp_filtered[:, :9])
            filtered_outputs.extend(tmp_filtered[:, 9:])

        original_inputs, original_outputs = np.asarray(original_inputs), np.asarray(original_outputs)
        filtered_inputs, filtered_outputs = np.asarray(filtered_inputs), np.asarray(filtered_outputs)
        original_outputs = np.c_[np.zeros([len(original_outputs), 3]), original_outputs[:, :-4], -np.mean(original_outputs[:, -4:-2], axis=1), -np.mean(original_outputs[:, -2:], axis=1)]
        filtered_outputs = np.c_[np.zeros([len(filtered_outputs), 3]), filtered_outputs[:, :-4], -np.mean(filtered_outputs[:, -4:-2], axis=1), -np.mean(filtered_outputs[:, -2:], axis=1)]

        return original_inputs, original_outputs, filtered_inputs, filtered_outputs






