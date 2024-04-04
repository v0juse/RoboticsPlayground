# %% Importing libraries
import sympy as sp


# %% Defining rotation matrices
def Rx(gamma, radians=True):
    """
    Rotation matrix about the x-axis.
    param gamma: angle of rotation
    """
    if not radians:
        gamma = sp.rad(gamma)
    return sp.Matrix(
        [
            [1, 0, 0],
            [0, sp.cos(gamma), -sp.sin(gamma)],
            [0, sp.sin(gamma), sp.cos(gamma)],
        ]
    )


def Ry(beta, radians=True):
    if not radians:
        beta = sp.rad(beta)
    return sp.Matrix(
        [
            [sp.cos(beta), 0, sp.sin(beta)],
            [0, 1, 0],
            [-sp.sin(beta), 0, sp.cos(beta)],
        ]
    )


def Rz(alpha, radians=True):
    if not radians:
        alpha = sp.rad(alpha)
    return sp.Matrix(
        [
            [sp.cos(alpha), -sp.sin(alpha), 0],
            [sp.sin(alpha), sp.cos(alpha), 0],
            [0, 0, 1],
        ]
    )


# %% Homogenous transformation matrix
def homogenous_matrix(R: sp.Matrix, t: sp.Matrix) -> sp.Matrix:
    """Homogenous transformation matrix.

    Args:
        R (sp.Matrix): Rotation portion of the matrix
        t (sp.Matrix): Translation portion of the matrix

    Returns:
        sp.Matrix: Homogenous transformation matrix
    """
    return sp.Matrix(
        [
            [R[0, 0], R[0, 1], R[0, 2], t[0]],
            [R[1, 0], R[1, 1], R[1, 2], t[1]],
            [R[2, 0], R[2, 1], R[2, 2], t[2]],
            [0, 0, 0, 1],
        ]
    )


# %% Testing matrix multiplication
# TODO: Abstract this to a function
denavit_hartenberg = [
    {"theta": sp.symbols("theta1"), "d": 0, "a": 0, "alpha": -sp.pi / 2},
    {"theta": sp.symbols("theta2"), "d": sp.symbols("d2"), "a": 0, "alpha": sp.pi / 2},
    {"theta": 0, "d": sp.symbols("d3"), "a": 0, "alpha": 0},
]
A = []
for i, dh_parameters in enumerate(denavit_hartenberg):
    theta, d, a, alpha = dh_parameters.values()
    print(f"DH parameters for line {i+1}: {theta, d, a, alpha}")
    A.append(
        homogenous_matrix(Rz(theta), sp.Matrix([0, 0, d]))
        * homogenous_matrix(Rx(alpha), sp.Matrix([a, 0, 0]))
    )
A_result = A[0]
for i, a in enumerate(A):
    if i == 0:
        continue
    A_result = A_result * a
sp.simplify(A_result)
# theta, d, a, alpha = sp.symbols("theta d a alpha")
# homogenous_matrix(Rz(theta), sp.Matrix([0, 0, d])) * homogenous_matrix(
# Rx(alpha), sp.Matrix([a, 0, 0])
# )
