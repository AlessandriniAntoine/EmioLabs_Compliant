import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

####################################################################################
# Note: This is where you implement the identification of the reduced linear model.
####################################################################################
def identify_model(states, inputs, outputs, forces, reduced_positions, r):
    x_next = states[:, 1:]
    x_curr = states[:, :-1]
    u_curr = inputs[:, :-1]
    f_curr = forces[:, :-1]

    state_input_force = np.vstack((x_curr, u_curr, f_curr))  # Shape: (nx + nu + nf, L - 1)
    ABE = x_next @ np.linalg.pinv(state_input_force)  # Shape: (nx, nx + nu + nf)

    A = ABE[:, :states.shape[0]]  # State matrix A
    B = ABE[:, states.shape[0]:states.shape[0] + inputs.shape[0]]  # Input matrix B
    E = ABE[:, states.shape[0] + inputs.shape[0]:]  # Force matrix E

    Cp = outputs @ np.linalg.pinv(reduced_positions)  # Shape: (ny, np)
    C = np.hstack((np.zeros_like(Cp), Cp))  # Shape: (ny, nx)

    return A, B, E, C

####################################################################################
# Note: The following code is provided and loads data, handles paths, and runs logic.
####################################################################################
lab_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_path = os.path.join(lab_path, "data")


def get_parser_args():
    parser = argparse.ArgumentParser(description='Identify a reduced state-space model.')
    parser.add_argument('--order', type=int, default=6,
        help="Reduction order used to build the model")
    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([])
    return args

def perform_identification():
    args = get_parser_args()

    # Load recorded data
    openloop_path = os.path.join(data_path, "sofa_openLoop.npz")
    if not os.path.exists(openloop_path):
        raise FileNotFoundError(f"Data file not found at: {openloop_path}. Please run the open-loop simulation first.")
    data = np.load(openloop_path)
    legs_pos = data["legPos"].T
    legs_vel = data["legVel"].T
    states_full = np.vstack((legs_vel, legs_pos))  # Full state (v; p)
    outputs = data["markersPos"].T
    inputs = data["motorPos"].T
    forces = data["force"].T

    # Load reduction matrix
    reduction_file = os.path.join(data_path, f"reduction_order{args.order}.npz")
    if not os.path.exists(reduction_file):
        raise FileNotFoundError(f"Reduction file not found at: {reduction_file}. Please run the reduction script first.")
    red = np.load(reduction_file)
    T = red["reductionMatrixPos"]
    R = red["reductionMatrix"]

    # Reduce states and positions
    reduced_states = R.T @ states_full
    reduced_positions = T.T @ legs_pos

    # Identify the model
    A, B, E, C = identify_model(reduced_states, inputs, outputs, forces, reduced_positions, args.order)

    # Print dimensions
    print(f"System matrices:\nA: {A.shape}, B: {B.shape}, E: {E.shape}, C: {C.shape}")

    # === Optional: Predict and simulate model ===
    L = reduced_states.shape[1]
    nx = A.shape[0]

    # Prediction using known data
    x_pred = A @ reduced_states + B @ inputs + E @ forces
    y_pred = C @ x_pred

    # Simulation from initial condition
    x_sim = reduced_states[:, 0].reshape(-1, 1)
    X_sim = np.zeros_like(reduced_states)
    Y_sim = np.zeros_like(outputs)
    X_sim[:, 0] = x_sim[:, 0]
    Y_sim[:, 0] = (C @ x_sim)[:, 0]
    for k in range(1, L):
        u = inputs[:, k-1].reshape(-1, 1)
        f = forces[:, k-1].reshape(-1, 1)
        x_sim = A @ x_sim + B @ u + E @ f
        X_sim[:, k] = x_sim[:, 0]
        Y_sim[:, k] = (C @ x_sim)[:, 0]

    # Plot results
    fig, axes = plt.subplots(nx, 1, figsize=(6, 2*nx))
    for i in range(nx):
        axes[i].plot(reduced_states[i], label="Original", color="red")
        axes[i].plot(x_pred[i], label="Prediction", linestyle="--", color="blue")
        axes[i].plot(X_sim[i], label="Simulation", linestyle="--", color="green")
        axes[i].set_ylabel(f"$x_{i}$")
    axes[0].legend()
    fig.suptitle("Reduced States")

    ny = outputs.shape[0]
    fig, axes = plt.subplots(ny, 1, figsize=(6, 2*ny))
    for i in range(ny):
        axes[i].plot(outputs[i], label="Original", color="red")
        axes[i].plot(y_pred[i], label="Prediction", linestyle="--", color="blue")
        axes[i].plot(Y_sim[i], label="Simulation", linestyle="--", color="green")
        axes[i].set_ylabel(f"$y_{i}$")
    axes[0].legend()
    fig.suptitle("Outputs")
    plt.show()

    # Save model
    np.savez(os.path.join(data_path, f"model_order{args.order}.npz"),
             stateMatrix=A,
             inputMatrix=B,
             forceMatrix=E,
             outputMatrix=C)

# Optional SOFA interface
def createScene(root):
    perform_identification()

# Entry point
if __name__ == "__main__":
    perform_identification()
