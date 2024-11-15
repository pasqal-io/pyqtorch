from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import qutip
from scipy.interpolate import UnivariateSpline
from whittaker_eilers import WhittakerSmoother

# user parameters
SMOOTH_LMBDA = 5e-2
SMOOTH_ORDER = 2
INTERPOLATE_K = 5
INTERPOLATE_S = 0.01

# parameters and device specs
PATH = "EXPERIMENTS/2024-fresnel-dqc-qel-rerun/agpsr-publication/"
THETA3S = [
    0.70,
    1.40,
    2.09,
    2.79,
    3.49,
    4.19,
    4.88,
    5.58,
    6.28,
]
QEL_THETA3S_IDX = 3
PULSE_AMPLITUDE = 9
MIN_DURATION = 16
SHIFTS = np.array([2.4699029, 0.90160249])
X_POINTS = np.linspace(
    MIN_DURATION * PULSE_AMPLITUDE / 1000 + max(SHIFTS),
    MIN_DURATION * PULSE_AMPLITUDE / 1000 + max(SHIFTS) + 5,
    8,
)  # points at which derivatives are to be calculated
QEL_X_POINTS = [
    4.5186648,
    4.99485528,
    5.23295051,
    5.709141,
    5.94723624,
]  # additional points at which derivatives are to be calculated when θ_3=2.79
C6 = 865723.02
DURATION_TO_ANGLE = PULSE_AMPLITUDE / 1000


# generator corresponding to a global pulse, specialized for two atoms 5*sqrt(3)um apart
def multi_qubit_rotation_generator(omega: float):
    n_qubits = 2

    # create rotation operators
    op_A = []
    for i in range(n_qubits):
        prod = [qutip.qeye(2) for _ in range(n_qubits)]
        prod[i] = omega / 2 * qutip.sigmax()
        op_A.append(qutip.tensor(prod))
    A = sum(op_A)

    # create interaction operators
    prod = [(1 + qutip.sigmaz()) / 2, (1 + qutip.sigmaz()) / 2]
    r_ij = 5 * np.sqrt(3)
    J_ij = C6 / (r_ij**6)
    B = J_ij * qutip.tensor(prod)

    # construct generator
    G = (A + B) * (2.0 / omega)

    return G


def plot_all():
    # SETUP FOR GPSR
    # find unique eigenvalue differences
    _G = multi_qubit_rotation_generator(omega=PULSE_AMPLITUDE)
    print("Generator :", _G)
    _eigenvals = _G.eigenenergies()
    _diffs = np.round(_eigenvals - _eigenvals.reshape(-1, 1), 7)
    _unique_diffs = np.abs(np.unique(np.tril(_diffs)))
    _unique_diffs = _unique_diffs[_unique_diffs > 0]
    print("Unique eigenvalues differences:", _unique_diffs)
    print(
        f"Shifts {SHIFTS} used to obtain QPU data was optimized for [4.032524  1.9007948]"
    )
    UNIQUE_EVAL_DIFFS = np.array([_unique_diffs[0], _unique_diffs[-2]])
    print("Eigenvalues differences used:", UNIQUE_EVAL_DIFFS)
    # calculate M_INV matrix
    _M = np.empty((2, 2))
    for i in range(2):
        for j in range(2):
            _M[i, j] = 4 * np.sin(SHIFTS[i] * UNIQUE_EVAL_DIFFS[j] / 2)
    M_INV = np.linalg.pinv(_M)

    # PLOT FOR EACH θ_3 VALUE
    fig, _ = plt.subplots(3, 3, sharex=True)
    for i in range(len(THETA3S)):
        theta3 = THETA3S[i]
        if theta3 == THETA3S[QEL_THETA3S_IDX]:
            x_pts_for_derivative = np.sort(np.hstack((X_POINTS, QEL_X_POINTS)))
        else:
            x_pts_for_derivative = X_POINTS

        # LOAD DATA
        with open(f"{PATH}data_{theta3}.json", "r") as file:
            json_data = json.load(file)
        data = np.array(
            [
                [item["duration"] * DURATION_TO_ANGLE, item["f"], item["err_f"]]
                for item in json_data
            ]
        )

        # GPSR
        dfdx_GPSR = []

        def closest_x_index(x: float):
            difference_array = np.absolute(data[:, 0] - x)
            index = difference_array.argmin()
            # print(difference_array[index])
            return index

        for x_v in x_pts_for_derivative:
            f1u = data[closest_x_index(x_v + SHIFTS[0]), 1]
            f1l = data[closest_x_index(x_v - SHIFTS[0]), 1]
            f2u = data[closest_x_index(x_v + SHIFTS[1]), 1]
            f2l = data[closest_x_index(x_v - SHIFTS[1]), 1]
            F = np.array([f1u - f1l, f2u - f2l])
            dfdx_v = np.sum(UNIQUE_EVAL_DIFFS * np.dot(M_INV, F))
            dfdx_GPSR.append(dfdx_v)

        whittaker_smoother = WhittakerSmoother(
            lmbda=SMOOTH_LMBDA,
            order=SMOOTH_ORDER,
            data_length=len(data[:, 1]),
            x_input=data[:, 0],
        )
        smoothed_data = whittaker_smoother.smooth(data[:, 1])
        interpolation = UnivariateSpline(
            data[:, 0], smoothed_data, k=INTERPOLATE_K, s=INTERPOLATE_S
        )
        derivative = interpolation.derivative()

        ax = fig.axes[i]
        ax.errorbar(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            label="Experiment",
            color="tab:blue",
            linestyle="none",
            marker="o",
            ms=2,
        )
        ax.plot(
            data[:, 0],
            smoothed_data,
            label="Smoothed",
            color="tab:green",
            marker="o",
            ms=4,
        )
        ax.plot(
            x_pts_for_derivative,
            dfdx_GPSR,
            label="GPSR",
            color="tab:orange",
            marker="o",
            ms=4,
        )
        x_v = np.linspace(min(data[:, 0]), max(data[:, 0]), 300)
        x_v_derivative = np.linspace(min(X_POINTS), max(X_POINTS), 200)
        ax.plot(
            x_v,
            interpolation(x_v),
            label="Interpolation",
            color="tab:red",
        )
        ax.plot(
            x_v_derivative,
            derivative(x_v_derivative),
            label="Derivative of interpolation",
            color="tab:pink",
        )

        ax.text(
            0.01,
            0.99,
            "$θ_3$ = " + f"{theta3}",
            ha="left",
            va="top",
            transform=ax.transAxes,
        )

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels)
    plt.show()


if __name__ == "__main__":
    plot_all()
