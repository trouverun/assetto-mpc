import json
import config
import torch
import numpy as np
import os
from utils.general import generate_randomly_spaced_indices
from dynamics_identification.torch_dynamics_models.single_track_bicycle import SingleTrackBicycle
from dynamics_identification.torch_dynamics_models.gaussian_process import GPModel

if __name__ == '__main__':
    result_folder = "dynamics_identification/gp_training"
    os.makedirs(result_folder, exist_ok=True)

    data = []
    source_dirs = []
    ROOT_DIR = "C:\\driving_data"
    for folder in os.listdir(ROOT_DIR):
        for file in os.listdir("%s/%s" % (ROOT_DIR, folder)):
            if ".npy" in file:
                source_dirs.append("%s/%s" % (ROOT_DIR, folder))
                temp = np.load("%s/%s/%s" % (ROOT_DIR, folder, file))
                print(temp.shape)
                print(file)
                print(temp['Pitch'])
                valid = np.all([temp['Is_Valid_Lap'], np.logical_not(temp['Is_In_Pit']), temp['Local_Velocity_Z'] > 7, np.abs(temp['Pitch']) < 0.085, np.abs(temp['Roll']) < 0.085], axis=0)
                dvalid = valid[1:].astype(int) - valid[:-1].astype(int)
                stops = np.argwhere(dvalid != 0)[:, 0]
                if not len(stops):
                    data.append(temp)
                prev = 0
                for stop in stops:
                    if dvalid[stop] == -1:
                        if stop - prev > 1000:
                            data.append(temp[prev:stop])
                    prev = stop
                if prev < len(temp) - 1 and (len(temp) - 1) - prev > 1000:
                    data.append(temp[prev:])

    torch_model = SingleTrackBicycle().cuda()
    print("EXTRACTING")
    _, _, filtered_inputs, filtered_outputs = torch_model.extract_inputs_outputs(data, source_dirs)
    print(len(filtered_inputs))

    bike_inputs = torch.from_numpy(filtered_inputs).to(torch.float).cuda()
    bike_targets = filtered_outputs[:, 3:6]
    print(np.amin(bike_targets, axis=0))
    print(np.amax(bike_targets, axis=0))
    gp_inputs = torch.from_numpy(filtered_inputs[:, 1:6]).to(torch.float).cuda()
    gp_targets = torch.from_numpy(filtered_outputs[:, 3:6]).to(torch.float).cuda()

    with torch.no_grad():
        bicycle_outputs = torch_model(bike_inputs)[:, 3:6]
    gp_targets -= bicycle_outputs

    def farthest_point_sampling(points, num_samples, print_interval=5000):
        points = torch.from_numpy(points).cuda()
        num_points = points.shape[0]

        # Start with a random point
        first_index = torch.randint(num_points, (1,)).item()
        sampled_indices = [first_index]

        # Initialize the distance list with distances to the first point
        min_distances = torch.linalg.vector_norm(points - points[first_index], dim=1)

        for i in range(1, num_samples):
            if i % print_interval == 0:
                print(f"Progress: {i}/{num_samples} samples selected.")

            # Get the farthest point based on current min_distances
            farthest_point_idx = torch.argmax(min_distances).item()
            sampled_indices.append(farthest_point_idx)

            # Update the minimum distances with respect to the new farthest point
            distances_to_new_point = torch.linalg.vector_norm(points - points[farthest_point_idx], dim=1)
            min_distances = torch.min(min_distances, distances_to_new_point)

        return sampled_indices


    stacked_data = np.c_[bike_inputs.cpu().numpy()[:, 1:6], bike_targets]
    selected = farthest_point_sampling(gp_inputs.cpu().numpy() / [60, 15, 1, 1, 1.5], int(2e4))
    # selected = np.random.choice(np.arange(len(gp_inputs)), int(min(2e4, len(gp_inputs))), replace=False)
    print(len(selected))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(10000 / 100, 10000 / 100))
    fig.tight_layout()

    for i in range(2):
        for j in range(2):
            if i == 0:
                if j == 0:
                    ax[i, j].set_title("linear velocities")
                    ax[i, j].scatter(stacked_data[selected, 0], stacked_data[selected, 1])
                if j == 1:
                    ax[i, j].set_title("linear accelerations")
                    ax[i, j].scatter(stacked_data[selected, 5], stacked_data[selected, 6])

                    axmin = 16
                    axmax = 9
                    aymax = 20

                    vertices = [
                        (-axmin, 0),
                        (0, -aymax),
                        (axmax, 0),
                        (0, aymax)
                    ]

                    # Plotting
                    polygon = plt.Polygon(vertices, fill=None, edgecolor='r', alpha=1)
                    ax[i, j].add_patch(polygon)

                    xs = 7.5  # Half-length along x-axis
                    ys = 15  # Half-length along y-axis
                    # Create theta values
                    theta = np.linspace(0, 2 * np.pi, 100)

                    # Compute x and y values for the ellipse
                    x = xs * np.cos(theta)
                    y = ys * np.sin(theta)
                    ax[i, j].fill(x-2, y, alpha=0.5)  # This will fill the interior of the ellipse

            if i == 1:
                if j == 0:
                    ax[i, j].set_title("angulars")
                    ax[i, j].scatter(stacked_data[selected, 2], stacked_data[selected, 7])
                if j == 1:
                    ax[i, j].set_title("controls")
                    ax[i, j].scatter(stacked_data[selected, 3], stacked_data[selected, 4])
    plt.savefig(f"{result_folder}/samples.png")
    plt.close()

    input_scaler = torch.tensor([config.lin_vel_acc_scaler, config.lin_vel_acc_scaler, 1., 1., 1.]).cuda()
    output_scaler = torch.tensor([config.lin_vel_acc_scaler, config.lin_vel_acc_scaler, 1.]).cuda()

    gp_model = GPModel(gp_inputs[selected] / input_scaler, gp_targets[selected] / output_scaler, train_epochs=25, train_lr=1e-1)

    params = {k: v.cpu().numpy().tolist() for k, v in gp_model.state_dict().items()}
    with open(f"{result_folder}/gp_params.json", "w") as outfile:
        outfile.write(json.dumps(params, indent=4))

    eval_indices = generate_randomly_spaced_indices(len(filtered_inputs), 1000, 20)
    eval_indices = np.asarray(eval_indices).flatten()
    bike_targets = bike_targets[eval_indices]
    bicycle_outputs = bicycle_outputs[eval_indices]
    gp_inputs = gp_inputs[eval_indices]

    with torch.no_grad():
        gp_predictions = gp_model(gp_inputs / input_scaler)

    titles = [
        ("ax", 0, False),
        ("ay", 1, False),
        ("dw", 2, False),
    ]
    fig, ax = plt.subplots(len(titles), 1, figsize=(10000 / 100, 10000 / 100))
    fig.tight_layout()
    for j, (title, idx, is_input) in enumerate(titles):
        ax[j].set_title(title)
        scaler = 1
        if title in ['hdg', 'w', 'dw']:
            scaler = 180 / np.pi
        ax[j].plot(scaler * bike_targets[:, idx], label="truth")
        ax[j].plot(scaler * bicycle_outputs[:, idx].cpu().numpy(), label="bicycle")
        mean = gp_predictions.mean[:, idx] * output_scaler[idx]
        combined = scaler * (bicycle_outputs[:, idx].cpu().numpy() + mean.cpu().numpy())
        ax[j].plot(combined, label="bicycle + gp")
        ax[j].fill_between(np.arange(len(combined)),
            combined - scaler * 2*gp_predictions.variance.sqrt()[:, idx].cpu().numpy(),
            combined + scaler * 2*gp_predictions.variance.sqrt()[:, idx].cpu().numpy(),
            alpha=0.5)

        ax[j].legend()
    plt.savefig(f"{result_folder}/fit.png")
    plt.close()

