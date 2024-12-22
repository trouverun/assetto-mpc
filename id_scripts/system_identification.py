import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from utils.general import generate_randomly_spaced_indices
from dynamics_identification.torch_dynamics_models.single_track_bicycle import SingleTrackBicycle
from utils.dynamics_trainer import EvoTorchWrapper


if __name__ == "__main__":
    output_folder = "dynamics_identification/id_results"
    os.makedirs(output_folder, exist_ok=True)

    data = []
    source_dirs = []
    ROOT_DIR = "C:\\driving_data"
    for folder in os.listdir(ROOT_DIR):
        for file in os.listdir("%s/%s" % (ROOT_DIR, folder)):
            if ".npy" in file:
                source_dirs.append("%s/%s" % (ROOT_DIR, folder))
                temp = np.load("%s/%s/%s" % (ROOT_DIR, folder, file))
                valid = np.all([temp['Is_Valid_Lap'], np.logical_not(temp['Is_In_Pit']), temp['Local_Velocity_Z'] > 7], axis=0)
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
                if prev < len(temp)-1 and (len(temp)-1) - prev > 1000:
                    data.append(temp[prev:])

    torch_model = SingleTrackBicycle()
    original_inputs, original_outputs, filtered_inputs, filtered_outputs = torch_model.extract_inputs_outputs(data, source_dirs)

    evo = EvoTorchWrapper(torch.from_numpy(filtered_inputs), torch.from_numpy(filtered_outputs), torch_model)
    evo.solve(popsize=250, iters=250, std=0.25)

    eval_indices = generate_randomly_spaced_indices(len(filtered_inputs), 1000, 20)
    eval_indices = np.asarray(eval_indices).flatten()
    original_inputs = original_inputs[eval_indices]
    original_outputs = original_outputs[eval_indices]
    filtered_inputs = filtered_inputs[eval_indices]
    filtered_outputs = filtered_outputs[eval_indices]

    filtered_inputs = torch.from_numpy(filtered_inputs).to(torch.float).cuda()
    with torch.no_grad():
        model_outputs = torch_model(filtered_inputs).cpu().numpy()
    filtered_inputs = filtered_inputs.cpu().numpy()

    params = {k: v.item() for k, v in torch_model.state_dict().items()}
    with open("%s/dynamics_params/bicycle_params.json" % output_folder, "w") as outfile:
        outfile.write(json.dumps(params, indent=4))

    rates = np.linspace(-1, 1, 100)
    Fry = params["wheel_Dr"] * np.sin(params["wheel_Cr"] * np.arctan(params["wheel_Br"] * rates))
    Ffy = params["wheel_Df"] * np.sin(params["wheel_Cf"] * np.arctan(params["wheel_Bf"] * rates))
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    ax[0].plot(rates, Fry)
    ax[0].title.set_text('Rear tire forces')
    ax[1].plot(rates, Ffy)
    ax[1].title.set_text('Front tire forces')
    plt.savefig(output_folder + "/tire_forces.png")

    titles = [
        ("velocities", [1, 2], True),
        ("w", 3, True),
        ("steer", 4, True),
        ("throttle", 5, True),
        ("dsteer", 6, True),
        ("ax", 3, False),
        ("ay", 4, False),
        ("dw", 5, False),
        ("fslip", 6, False),
        ("rslip", 7, False)
    ]

    fig, ax = plt.subplots(len(titles), 1, figsize=(10000/100, 10000/100))
    fig.tight_layout()
    for j, (title, idx, is_input) in enumerate(titles):
        ax[j].set_title(title)
        scaler = 1
        if title in ['hdg', 'w', 'dw']:
            scaler = 180 / np.pi
        if is_input:
            if isinstance(idx, list):
                for i in idx:
                    ax[j].plot(scaler * original_inputs[:, i], label="raw data")
                    ax[j].plot(scaler * filtered_inputs[:, i], label="filtered data")
                ax[j].plot(np.zeros_like(filtered_inputs[:, 0]))
            else:
                ax[j].plot(scaler * original_inputs[:, idx], label="raw data")
                ax[j].plot(scaler * filtered_inputs[:, idx], label="filtered (training) data")
        else:
            ax[j].plot(scaler * original_outputs[:, idx], label="raw data")
            ax[j].plot(scaler * filtered_outputs[:, idx], label="filtered (training) data")
            ax[j].plot(scaler * model_outputs[:, idx], label="model output")
            if idx == 4:
                ax[j].plot(scaler * filtered_inputs[:, 3] * filtered_inputs[:, 1], label="wvy")

        ax[j].legend()
    plt.savefig(output_folder + "/bicycle_fit.png")
    plt.close()
