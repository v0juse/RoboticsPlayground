# %% Importing libraries
import sympy as sp
import pydantic as pyd


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
def homogenous_matrix(
    Rot: sp.Matrix = sp.Identity(3), trans: sp.Matrix = [0, 0, 0]
) -> sp.Matrix:
    """Homogenous transformation matrix.

    Args:
        R (sp.Matrix): Rotation portion of the matrix
        t (sp.Matrix): Translation portion of the matrix

    Returns:
        sp.Matrix: Homogenous transformation matrix
    """
    return sp.Matrix(
        [
            [Rot[0, 0], Rot[0, 1], Rot[0, 2], trans[0]],
            [Rot[1, 0], Rot[1, 1], Rot[1, 2], trans[1]],
            [Rot[2, 0], Rot[2, 1], Rot[2, 2], trans[2]],
            [0, 0, 0, 1],
        ]
    )


# %% Testing matrix multiplication

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


# %% Denavid-Hartenberg class
class DenavitHartenbergParams(pyd.BaseModel):
    class Config:
        arbitrary_types_allowed = True

    theta: sp.Symbol | float
    d: sp.Symbol | float
    a: sp.Symbol | float
    alpha: sp.Symbol | float


class FromDenavidHatenberg:
    def __init__(self, dh_parameters: list[DenavitHartenbergParams]):
        self.dh_parameters = dh_parameters
        self.A, self.T_all = self.get_homogenous_matrix()

    def get_homogenous_matrix(self):
        A = []
        for i, dh_parameters in enumerate(self.dh_parameters):
            theta, d, a, alpha = dh_parameters.values()
            print(f"DH parameters for line {i+1}: {theta, d, a, alpha}")
            A.append(
                homogenous_matrix(Rz(theta), sp.Matrix([0, 0, d]))
                * homogenous_matrix(Rx(alpha), sp.Matrix([a, 0, 0]))
            )
        T_all = A[0]
        for i, a in enumerate(A):
            if i == 0:
                continue
            T_all = T_all * a
        return A, sp.simplify(T_all)


# %% Spherical manipulator
spherical_dh_params: list[DenavitHartenbergParams] = [
    {"theta": sp.symbols("theta1"), "d": 0, "a": 0, "alpha": -sp.pi / 2},
    {"theta": sp.symbols("theta2"), "d": sp.symbols("d2"), "a": 0, "alpha": sp.pi / 2},
    {"theta": 0, "d": sp.symbols("d3"), "a": 0, "alpha": 0},
]
spherical_manipulator = FromDenavidHatenberg(spherical_dh_params)
spherical_manipulator.T_all

# %% Spherical wrist

spherical_wrist_dh_params: list[DenavitHartenbergParams] = [
    {"theta": sp.symbols("theta1"), "d": 0, "a": 0, "alpha": -sp.pi / 2},
    {"theta": sp.symbols("theta2"), "d": 0, "a": 0, "alpha": sp.pi / 2},
    {"theta": sp.symbols("theta3"), "d": 0, "a": 0, "alpha": 0},
]
spherical_wrist = FromDenavidHatenberg(spherical_wrist_dh_params)
spherical_wrist.T_all

# %% Inverse kinematics
phi = 45
theta = 0
psi = 90
R = Rz(phi, radians=False) * Ry(theta, radians=False) * Rx(psi, radians=False)
R_homogeneus = homogenous_matrix(Rot=R)

spherical_wrist.T_all


# FIXME system solving not working
def solve_inverse_kinematics(
    manipulator_matrix: sp.Matrix, desired_matrix: sp.Matrix, n_equations=3
):
    zero_matrix = manipulator_matrix - desired_matrix
    print(zero_matrix)
    return sp.nsolve(
        (
            zero_matrix[2, 0],
            zero_matrix[2, 1],
            zero_matrix[2, 2],
            zero_matrix[3, 0],
            zero_matrix[3, 1],
            zero_matrix[3, 2],
        ),
        (sp.symbols("theta1"), sp.symbols("theta2"), sp.symbols("theta3")),
        (0, 0, 0),
    )


solve_inverse_kinematics(spherical_wrist.T_all, R_homogeneus)
# theta, d, a, alpha = sp.symbols("theta d a alpha")
# homogenous_matrix(Rz(theta), sp.Matrix([0, 0, d])) * homogenous_matrix(
# Rx(alpha), sp.Matrix([a, 0, 0])
# )
