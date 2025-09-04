import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import control as ct

####################################################################################
# Note: This is where you design the state feedback controller and compute G.
####################################################################################
def design_controller(A, B, C):
    ctrb_rank = np.linalg.matrix_rank(ct.ctrb(A, B))
    if ctrb_rank != A.shape[0]:
        raise ValueError(f"System is not controllable: rank {ctrb_rank}, expected {A.shape[0]}.")
    C_ctr = C[[1]]
    if C_ctr.shape[0] != B.shape[1]:
        raise ValueError(f"Controlled output must have the same dimension as the input. Got {C_ctr.shape[0]} controlled outputs and {B.shape[1]} inputs.")

    Q = C.T @ (1e-1 * np.eye(C.shape[0])) @ C
    R = 1e3 * np.eye(B.shape[1])
    K, _, _ = ct.dlqr(A, B, Q, R)
    G_inv = (C_ctr @ np.linalg.inv(np.eye(A.shape[0]) - A + B @ K) @ B)
    G = np.linalg.pinv(G_inv)

    eig_open = np.linalg.eigvals(A)
    print(f"Open-loop eigenvalue magnitudes: {np.abs(eig_open)}")
    eig_closed = np.linalg.eigvals(A - B @ K)
    print(f"Closed-loop eigenvalue magnitudes: {np.abs(eig_closed)}")
    return K, G

def design_integral_controller(A, B, C):
    C_ctr = C[[1]]
    Abar = np.block([[A, np.zeros((A.shape[0], B.shape[1]))], [-C_ctr, np.eye(B.shape[1])]])
    Bbar = np.block([[B], [np.zeros((B.shape[1], B.shape[1]))]])
    ctrb_rank = np.linalg.matrix_rank(ct.ctrb(Abar, Bbar))
    if ctrb_rank != Abar.shape[0]:
        raise ValueError(f"System is not controllable: rank {ctrb_rank}, expected {Abar.shape[0]}.")
    if C_ctr.shape[0] != B.shape[1]:
        raise ValueError(f"Controlled output must have the same dimension as the input. Got {C_ctr.shape[0]} controlled outputs and {B.shape[1]} inputs.")

    Q = np.block([[C.T @ (1e-0 * np.eye(C.shape[0])) @ C, np.zeros((A.shape[0], B.shape[1]))],
                  [np.zeros((B.shape[1], A.shape[0])), 1e-2 * np.eye(B.shape[1])]])
    R = 1e3 * np.eye(B.shape[1])
    K, _, _ = ct.dlqr(Abar, Bbar, Q, R)
    Kx = K[:, :A.shape[0]]
    Ki = K[:, A.shape[0]:]

    eig_open = np.linalg.eigvals(Abar)
    print(f"Open-loop eigenvalue magnitudes: {np.abs(eig_open)}")
    eig_closed = np.linalg.eigvals(Abar - Bbar @ K)
    print(f"Closed-loop eigenvalue magnitudes: {np.abs(eig_closed)}")
    print(f"State gain:\n{Kx}")
    print(f"Integral gain:\n{Ki}")
    return Kx, Ki

####################################################################################
# Note: The following code loads data, handles paths, and runs the controller logic.
####################################################################################
lab_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_path = os.path.join(lab_path, "data")


def get_parser_args():
    parser = argparse.ArgumentParser(description='Design a state feedback controller.')
    parser.add_argument('--order', type=int, default=6,
        help="Reduction order used for the model")
    parser.add_argument('--controller_type', type=str, default="state_feedback",
        help="Type of controller to design")
    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([])
    return args

def perform_controller_design():
    args = get_parser_args()

    # Load identified model
    model_file = os.path.join(data_path, f"model_order{args.order}.npz")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}. Please run identification first.")
    model = np.load(model_file)
    A = model["stateMatrix"]
    B = model["inputMatrix"]
    C = model["outputMatrix"]

    # Design observer
    if args.controller_type == "state_feedback_integral":
        Kx, Ki = design_integral_controller(A, B, C)
        G = np.zeros((B.shape[1], B.shape[1]))
    elif args.controller_type == "state_feedback":
        Kx, G = design_controller(A, B, C)
        Ki = np.zeros((B.shape[1], B.shape[1]))
    else:
        raise ValueError(f"Unknown controller type: {args.controller_type}")

    # Save controller
    np.savez(os.path.join(data_path, f"controller_order{args.order}.npz"),
             statefeedbackGain=Kx,
             integralfeedbackGain=Ki,
             feedforwardGain=G)

# Optional SOFA interface
def createScene(root):
    perform_controller_design()

# Entry point
if __name__ == "__main__":
    perform_controller_design()
