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

    # TODO: optimize this
    def __add__(self, other):
        return FromDenavidHatenberg(self.dh_parameters + other.dh_parameters)

    def add_base_transformation(self, Matrix: sp.Matrix):
        if hasattr(self, "base2zero"):
            raise ValueError("Base transformation already added for this robot.")
        self.base2zero = Matrix
        self.T_all = Matrix * self.T_all

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


# solve_inverse_kinematics(spherical_wrist.T_all, R_homogeneus)
# %% Spherical wrist

ex_quadro_dh_params: list[DenavitHartenbergParams] = [
    {"theta": sp.symbols("theta1"), "d": sp.symbols("d1"), "a": 0, "alpha": sp.pi / 2},
    {"theta": sp.symbols("theta2"), "d": 0, "a": 0, "alpha": sp.pi / 2},
    {"theta": 0, "d": sp.symbols("d3"), "a": 0, "alpha": 0},
]
ex_quadro = FromDenavidHatenberg(ex_quadro_dh_params)
ex_quadro.T_all

# homogenous_matrix(Rz(theta), sp.Matrix([0, 0, d])) * homogenous_matrix(
# Rx(alpha), sp.Matrix([a, 0, 0])
# )
# ========= Anthropomorphous manipulator =========
# %% Constructive parameters
a_0 = 0.5
b_0 = 0.5
a_2 = 0.85
d_4 = 0.57
d_1 = 0.82
d_6 = 0.17

alpha_1 = -sp.pi / 2
alpha_2 = 0
alpha_3 = -sp.pi / 2
alpha_4 = sp.pi / 2
alpha_5 = -sp.pi / 2
alpha_6 = 0
# %% Anthropomorphous manipulator

anthropomorphous_dh_params: list[DenavitHartenbergParams] = [
    {"theta": sp.symbols("theta1"), "d": d_1, "a": 0, "alpha": alpha_1},
    {"theta": sp.symbols("theta2"), "d": 0, "a": a_2, "alpha": alpha_2},
    {"theta": sp.symbols("theta3"), "d": 0, "a": 0, "alpha": alpha_3},
]
AnthropomorphousArm = FromDenavidHatenberg(anthropomorphous_dh_params)
# %% Spherical wrist
spherical_wrist_dh_params: list[DenavitHartenbergParams] = [
    {"theta": sp.symbols("theta4"), "d": d_4, "a": 0, "alpha": alpha_4},
    {"theta": sp.symbols("theta5"), "d": 0, "a": 0, "alpha": alpha_5},
    {"theta": sp.symbols("theta6"), "d": d_6, "a": 0, "alpha": alpha_6},
]
SphericalWrist = FromDenavidHatenberg(spherical_wrist_dh_params)

# %% Complete anthropomorphous manipulator with spherical wrist
Robot = AnthropomorphousArm + SphericalWrist
Robot.T_all

# %% Add base translation
base_translation = homogenous_matrix(trans=[a_0, b_0, 0])
Robot.add_base_transformation(base_translation)
# %% Testing the robot function with some values (all zero)
Robot.T_all.evalf(
    subs={
        sp.symbols("theta1"): 0,
        sp.symbols("theta2"): 0,
        sp.symbols("theta3"): 0,
        sp.symbols("theta4"): 0,
        sp.symbols("theta5"): 0,
        sp.symbols("theta6"): 0,
    }
)
# %% Testing the robot function with some values (not all zero)
Robot.T_all.evalf(
    subs={
        sp.symbols("theta1"): sp.pi / 4,
        sp.symbols("theta2"): -sp.pi / 4,
        sp.symbols("theta3"): 0,
        sp.symbols("theta4"): 0,
        sp.symbols("theta5"): sp.pi / 3,
        sp.symbols("theta6"): 0,
    }
)
# %% Testing anthropomorphous manipulator only
AnthropomorphousArm.add_base_transformation(base_translation)
AnthropomorphousArm.T_all.evalf(
    subs={
        sp.symbols("theta1"): 0,
        sp.symbols("theta2"): -sp.pi / 4,
        sp.symbols("theta3"): 0,
    }
)

# ==================== Inverse kinematics ====================
# %% Inverse kinematics
p_end = sp.Matrix([1.5, -0.1, 0.7])
n_end = sp.Matrix([1, 0, 0])
s_end = sp.Matrix([0, 1, 0])
a_end = sp.Matrix([0, 0, -1])

# %% 1st step wrist position
pw_end = p_end - d_6 * a_end
pw_end
# %% calculating joint angles for the wrist
# wrist is on O_4, using matrix T_4_0
T_b_wrist = Robot.base2zero * Robot.A[0] * Robot.A[1] * Robot.A[2] * Robot.A[3]
T_b_wrist 
# %% Solving the equations
solution = sp.nsolve(
(
        T_b_wrist[0, 3] - pw_end[0],
        T_b_wrist[1, 3] - pw_end[1],
        T_b_wrist[2, 3] - pw_end[2],  # fixed value
    ),
    (sp.symbols("theta1"), sp.symbols("theta2"), sp.symbols("theta3")),
    (0.1, 0.1, 0.1),  # Adjust the initial guess
    dict=True,
    bounds=[(-sp.pi/2, sp.pi/2), (-sp.pi/2, sp.pi/2), (-sp.pi/2, sp.pi/2)],
)
solution
# %% Testing position of the wrist
T_b_wrist.evalf(
    subs={
        sp.symbols("theta1"): solution[0][sp.symbols("theta1")],
        sp.symbols("theta2"): solution[0][sp.symbols("theta2")],
        sp.symbols("theta3"): solution[0][sp.symbols("theta3")],
    }
)
# %%
