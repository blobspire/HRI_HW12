import torch
import numpy as np
import os
import matplotlib

matplotlib.use("Agg") # Headless plotting for autoresearch optimization
from models import MLPPolicy
from main import dynamics
import matplotlib.pyplot as plt


# plot the trajectory the mobile robot executes
def plot_trajectory(xi, filename):
    plt.figure()
    plt.plot(xi[:,0], xi[:,1], 'bo-')
    plt.plot(xi[0,0], xi[0,1], 'ks')
    plt.plot(xi[0,2], xi[0,3], 'ro')
    plt.axis("equal")
    plt.savefig(filename)
    plt.close()

def load_model(model_number):
    model = MLPPolicy(state_dim=4, hidden_dim=64, action_dim=2)
    model.load_state_dict(torch.load("model_weights" + str(model_number), weights_only=True))
    model.eval()
    return model

def test_model(model_number, n_tests=500, seed=42):
    model = load_model(model_number)
    rng = np.random.default_rng(seed)
    errors = []

    for idx in range(n_tests):

        # reset the system
        state = rng.uniform(-10, 10, 4).astype(np.float32)

        # rollout the learned policy
        xi = [state]
        for _ in range(20):
            
            # get action from model
            state_tensor = torch.FloatTensor(state)
            action = model(state_tensor).detach().numpy()

            # compute next state
            state = dynamics(state, action)
            xi.append(state)
            
        # record the error
        errors.append(np.linalg.norm(state[0:2]-state[2:4]))

    # plot the final sampled trajectory for this model
    xi = np.array(xi)
    plot_trajectory(xi, "xi_" + str(model_number) + ".png")
    return np.array(errors)

if __name__ == "__main__":
    model_numbers = [5, 10, 20]
    results = []

    for model_number in model_numbers:
        errors = test_model(model_number)
        results.append(
            [
                "model_weights" + str(model_number),
                np.mean(errors),
                np.median(errors),
                np.min(errors),
                np.max(errors),
                np.std(errors),
            ]
        )

    print()
    print(f"{'Model':<16} {'Average':>9} {'Median':>9} {'Min':>9} {'Max':>9} {'Std Dev':>9}")
    print("-" * 66)
    for model_name, average, median, minimum, maximum, std_dev in results:
        print(
            f"{model_name:<16} "
            f"{average:>9.3f} "
            f"{median:>9.3f} "
            f"{minimum:>9.3f} "
            f"{maximum:>9.3f} "
            f"{std_dev:>9.3f}"
        )
