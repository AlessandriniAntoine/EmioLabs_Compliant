import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import control as ct

####################################################################################
# Note: This is where you design the observer gain matrix L.
####################################################################################
def design_observer_perturbation_force(A, B, E, C):
    Abar = np.block([
        [A, B, E],
        [np.zeros((B.shape[1], A.shape[0])), np.eye(B.shape[1]), np.zeros((B.shape[1], E.shape[1]))],
        [np.zeros((E.shape[1], A.shape[0])), np.zeros((E.shape[1], B.shape[1])), np.eye(E.shape[1])]
    ])
    Cbar = np.block([[C, np.zeros((C.shape[0], B.shape[1])), np.zeros((C.shape[0], E.shape[1]))]])

    obsv_rank = np.linalg.matrix_rank(ct.obsv(Abar, Cbar))
    if obsv_rank != Abar.shape[0]:
        raise ValueError(f"System is not observable: rank {obsv_rank}, expected {Abar.shape[0]}.")

    Q = np.diag(A.shape[0] * [1e2] + [1e0] + [1e4])
    R = 1e1 * np.eye(Cbar.shape[0])
    Lt, _, _ =  ct.dlqr(Abar.T, Cbar.T, Q, R)
    L = Lt.T

    eig_open = np.linalg.eigvals(Abar)
    print(f"Open-loop eigenvalue magnitudes: {np.abs(eig_open)}")
    eig_obs = np.linalg.eigvals(Abar - L @ Cbar)
    print(f"Observer eigenvalue magnitudes: {np.abs(eig_obs)}")
    return L[:A.shape[0]], L[A.shape[0]:A.shape[0]+B.shape[1]], L[A.shape[0]+B.shape[1]:]


def design_observer_force(A, E, C):
    Abar = np.block([
        [A, E],
        [np.zeros((E.shape[1], A.shape[0])), np.eye(E.shape[1])]
    ])
    Cbar = np.block([[C, np.zeros((C.shape[0], E.shape[1]))]])

    obsv_rank = np.linalg.matrix_rank(ct.obsv(Abar, Cbar))
    if obsv_rank != Abar.shape[0]:
        raise ValueError(f"System is not observable: rank {obsv_rank}, expected {Abar.shape[0]}.")

    Q = np.diag(A.shape[0] * [1e2] + [1e4])
    R = 1e1 * np.eye(Cbar.shape[0])
    Lt, _, _ =  ct.dlqr(Abar.T, Cbar.T, Q, R)
    L = Lt.T

    eig_open = np.linalg.eigvals(Abar)
    print(f"Open-loop eigenvalue magnitudes: {np.abs(eig_open)}")
    eig_obs = np.linalg.eigvals(Abar - L @ Cbar)
    print(f"Observer eigenvalue magnitudes: {np.abs(eig_obs)}")
    return L[:A.shape[0]], L[A.shape[0]:]


def design_observer_perturbation(A, B, C):
    Abar = np.block([
        [A, B],
        [np.zeros((B.shape[1], A.shape[0])), np.eye(B.shape[1])]
    ])
    Cbar = np.block([[C, np.zeros((C.shape[0], B.shape[1]))]])

    obsv_rank = np.linalg.matrix_rank(ct.obsv(Abar, Cbar))
    if obsv_rank != Abar.shape[0]:
        raise ValueError(f"System is not observable: rank {obsv_rank}, expected {Abar.shape[0]}.")

    Q = np.diag(A.shape[0] * [1e2] + [1e0])
    R = 1e1 * np.eye(Cbar.shape[0])
    Lt, _, _ =  ct.dlqr(Abar.T, Cbar.T, Q, R)
    L = Lt.T

    eig_open = np.linalg.eigvals(Abar)
    print(f"Open-loop eigenvalue magnitudes: {np.abs(eig_open)}")
    eig_obs = np.linalg.eigvals(Abar - L @ Cbar)
    print(f"Observer eigenvalue magnitudes: {np.abs(eig_obs)}")
    return L[:A.shape[0]], L[A.shape[0]:]


def design_observer(A, C):
    obsv_rank = np.linalg.matrix_rank(ct.obsv(A, C))
    if obsv_rank != A.shape[0]:
        raise ValueError(f"System is not observable: rank {obsv_rank}, expected {A.shape[0]}.")

    Q = np.diag(A.shape[0] * [1e2])
    R = 1e1 * np.eye(C.shape[0])
    Lt, _, _ =  ct.dlqr(A.T, C.T, Q, R)
    L = Lt.T

    eig_open = np.linalg.eigvals(A)
    print(f"Open-loop eigenvalue magnitudes: {np.abs(eig_open)}")
    eig_obs = np.linalg.eigvals(A - L @ C)
    print(f"Observer eigenvalue magnitudes: {np.abs(eig_obs)}")
    return L

####################################################################################
# Note: The following code loads data, handles paths, and runs the observer logic.
####################################################################################
lab_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_path = os.path.join(lab_path, "data")


def get_parser_args():
    parser = argparse.ArgumentParser(description='Design an observer.')
    parser.add_argument('--order', type=int, default=6,
        help="Reduction order used for the model")
    parser.add_argument('--observer_type', type=str, default="default",
        help="Reduction order used for the model")
    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([])
    return args

def perform_observer_design():
    args = get_parser_args()

    # Load identified model
    model_file = os.path.join(data_path, f"model_order{args.order}.npz")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}. Please run identification first.")
    model = np.load(model_file)
    A = model["stateMatrix"]
    B = model["inputMatrix"]
    E = model["forceMatrix"]
    C = model["outputMatrix"]

    # Design observer
    if args.observer_type == "force":
        Lx, Lf = design_observer_force(A, E ,C)
        Lp = np.zeros((B.shape[1], C.shape[0]))
    elif args.observer_type == "perturbation":
        Lx, Lp = design_observer_perturbation(A, B ,C)
        Lf = np.zeros((E.shape[1], C.shape[0]))
    elif args.observer_type == "perturbation_force":
        Lx, Lp, Lf = design_observer_perturbation_force(A, B, E,C)
    else:
        Lx = design_observer(A, C)
        Lp = np.zeros((B.shape[1], C.shape[0]))
        Lf = np.zeros((E.shape[1], C.shape[0]))

    # Save observer
    np.savez(os.path.join(data_path, f"observer_order{args.order}.npz"),
             stateGain=Lx,
             forceGain=Lf,
             perturbationGain=Lp,
    )

# Optional SOFA interface
def createScene(root):
    perform_observer_design()

# Entry point
if __name__ == "__main__":
    perform_observer_design()
