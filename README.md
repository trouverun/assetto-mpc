# assetto-mpc
This repository implements a nonlinear MPC, which uses the combination of a bicycle model and a sparse Gaussian Process (GP) to accurately model the nonlinear vehicle dynamics of a race car. The GP is used to correct any prediction errors in the bicycle model, which might arise due to varying tire properties or weather conditions. The training data of the GP is updated during execution to better mitigate these effects.

The combined dynamics model is used to formulate a stochastic MPC, which propagates the GP variance through the prediction horizon. As the position uncertainty of the car grows through the horizon, the track width is tightened by the same amount, as illustrated below:

![tube](https://github.com/user-attachments/assets/00c0edb4-309e-4ea0-9005-47594ea5b45a)

This improves the robustness to model uncertainty, and ensures that the car always remains within the race track boundaries. The constraint-tightening mechanism is paired with a contouring-control cost function, which incentivces fast progress through the race track. 

## Results
The implemented Uncertainty-Aware-MPC (UCA-MPC) was compared to a nominal MPC, which neglects the sparse GP dynamics and constraint tightening. The evaluation metric was the mean of 9 best lap times (out of 10 attempts) on the Ricardo Tormo circuit, recorded in ideal weather conditions. The results of this experiment are shown below:

![ideal](https://github.com/user-attachments/assets/ce939141-2d35-432c-a1b9-437a25295f1e)

The UCA-MPC is on average 2% faster than the nominal MPC, and 6.9% slower than the human lap record of 1:29:60. 

Next, taking advantage of the detailed simulation of ACC, the two controllers were evaluated in more challenging conditions, with varying amounts of rain. The evaluation metric was the fastest lap recorded in 10 attempts for each weather scenario, with results shown below. 

![challenging](https://github.com/user-attachments/assets/c4f26547-1a22-471e-b622-39cb6556d283)

Compared to the nominal MPC, which Did Not Finish (DNF) in any of its attempts, the UCA-MPC was able to record successful laps even in the medium rain weather conditions. This highlights the improved robustness against modelling uncertainty.
 

## See it in action (youtube)
[![See it in action](https://img.youtube.com/vi/SENTHq9ONTw/0.jpg)](https://www.youtube.com/watch?v=SENTHq9ONTw)


### How it works in practice
The car odometry information is read from a shared memory file which ACC writes to. The car odometry data is then used to construct the bicycle model car state for control, and to prepare new data for GP regression in an online manner. The MPC problem is solved using Acados, which needs to run in a WSL2 instance. GRPC nodes are used to establish low-latency communication between the main program and the Acados WSL2 node. The computed controls are fed back to the simulator using an emulated game controller. 

The high-level architecture is depicted below:

![Architecture](https://github.com/user-attachments/assets/dd84a2ae-e74c-4b4e-aef6-9c15b5f4dc89)
