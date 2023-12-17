import json

# Result collection
num_eval_laps = 3

# Rendering
map_resolution = 1280
map_distance = 175

try:
    with open('dynamics_identification/id_results/dynamics_params/bicycle_params.json', 'r') as infile:
        bicycle_params = json.load(infile)
except:
    raise Exception("No bicycle params, run system_identification.py first with steering dataset")


# Filtering
learning_low_pass_window = 100
low_pass_window = 1000
data_sample_time = 1/50
savgol_p = 1
savgol_k = 3
savgol_d_k = 3

# Learning
learning_delay = 0.1  # = n -> Send new GP training data every n seconds
patience = 50
pos_scaler = 100
lin_vel_acc_scaler = 25

# Control
mpc_sample_time = 1/20
mpc_N = 50
steer_max = 1
throttle_min = 1
throttle_max = 0.75
u_steer_max = 1.5
u_throttle_max = 3
# Assetto gear change state machine
gear_high_rpm = 6900
gear_low_rpm = 5550
gear_read_dt_ms = 50
gear_change_dt_ms = 500

KM_H = 1000/(60*60)
speed_limit = 250*KM_H
minimum_speed = 250*KM_H
max_speed_increment = 50*KM_H

nonuniform_sample_low = 1/10