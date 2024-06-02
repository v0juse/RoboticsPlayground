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

# =================================== Inverse kinematics =============================
# %% Inverse kinematics
p_end = sp.Matrix([1.5, -0.1, 0.7])
n_end = sp.Matrix([1, 0, 0])
s_end = sp.Matrix([0, -1, 0])
a_end = sp.Matrix([0, 0, -1])

# %% 1st step wrist position
pw_end = p_end - d_6 * a_end
pw_end
# %% calculating joint angles for the wrist
# wrist is on O_4, using matrix T_4_0
T_b_wrist = Robot.base2zero * Robot.A[0] * Robot.A[1] * Robot.A[2] * Robot.A[3]
T_b_wrist
# %% Solving the equations
wrist_position_solution = sp.nsolve(
    (
        T_b_wrist[0, 3] - pw_end[0],
        T_b_wrist[1, 3] - pw_end[1],
        T_b_wrist[2, 3] - pw_end[2],
    ),
    (sp.symbols("theta1"), sp.symbols("theta2"), sp.symbols("theta3")),
    (0.1, 0.1, 0.1),  # Adjust the initial guess
    dict=True,
    bounds=[(-sp.pi / 2, sp.pi / 2), (-sp.pi / 2, sp.pi / 2), (-sp.pi / 2, sp.pi / 2)],
)
wrist_position_solution
# %% Testing position of the wrist
T_b_wrist.evalf(
    subs={
        sp.symbols("theta1"): wrist_position_solution[0][sp.symbols("theta1")],
        sp.symbols("theta2"): wrist_position_solution[0][sp.symbols("theta2")],
        sp.symbols("theta3"): wrist_position_solution[0][sp.symbols("theta3")],
    }
)
# %% 2nd step wrist orientation
R_end = sp.Matrix([[*n_end], [*s_end], [*a_end]]).T
R_end
# %% calculating joint angles for the wrist
R_0_3 = Robot.A[0][:3, :3] * Robot.A[1][:3, :3] * Robot.A[2][:3, :3]
R_0_3_num = R_0_3.evalf(
    subs={
        sp.symbols("theta1"): wrist_position_solution[0][sp.symbols("theta1")],
        sp.symbols("theta2"): wrist_position_solution[0][sp.symbols("theta2")],
        sp.symbols("theta3"): wrist_position_solution[0][sp.symbols("theta3")],
    }
)

R_3_6_num = R_0_3_num.T * R_end
R_3_6_num

R_3_6 = Robot.A[3][:3, :3] * Robot.A[4][:3, :3] * Robot.A[5][:3, :3]
R_3_6
# %% Solving the equations


orientation_solution = sp.nsolve(
    (
        R_3_6[0, 2] - R_3_6_num[0, 2],
        R_3_6[1, 2] - R_3_6_num[1, 2],
        R_3_6[2, 2] - R_3_6_num[2, 2],
        R_3_6[0, 1] - R_3_6_num[0, 1],
        R_3_6[2, 0] - R_3_6_num[2, 0],
        R_3_6[1, 0] - R_3_6_num[1, 0],
    ),
    (sp.symbols("theta4"), sp.symbols("theta5"), sp.symbols("theta6")),
    (0.1, 0.1, 0.1),  # Adjust the initial guess
    dict=True,
    solver="bisect",
    bounds=[(-sp.pi / 2, sp.pi / 2), (-sp.pi / 2, sp.pi / 2), (-sp.pi / 2, sp.pi / 2)],
)
orientation_solution
# %% Testing orientation of the wrist

R_3_6.evalf(
    subs={
        sp.symbols("theta4"): orientation_solution[0][sp.symbols("theta4")],
        sp.symbols("theta5"): orientation_solution[0][sp.symbols("theta5")],
        sp.symbols("theta6"): orientation_solution[0][sp.symbols("theta6")],
    }
), R_3_6_num
# %% Full solution
end_position_joints = {**wrist_position_solution[0], **orientation_solution[0]}
end_position_joints
# %% Evaluate the full solution
Robot.T_all.evalf(
    subs={
        sp.symbols("theta1"): end_position_joints[sp.symbols("theta1")],
        sp.symbols("theta2"): end_position_joints[sp.symbols("theta2")],
        sp.symbols("theta3"): end_position_joints[sp.symbols("theta3")],
        sp.symbols("theta4"): end_position_joints[sp.symbols("theta4")],
        sp.symbols("theta5"): end_position_joints[sp.symbols("theta5")],
        sp.symbols("theta6"): end_position_joints[sp.symbols("theta6")],
    }
)


# %% Extracting the solution to a function
def reverse_kinematics(
    d_6: float,
    Robot: FromDenavidHatenberg,
    p_end: sp.Matrix,
    n_end: sp.Matrix,
    s_end: sp.Matrix,
    a_end: sp.Matrix,
) -> dict:
    pw_end = p_end - d_6 * a_end
    # wrist is on O_4, using matrix T_4_0
    T_b_wrist = Robot.base2zero * Robot.A[0] * Robot.A[1] * Robot.A[2] * Robot.A[3]
    wrist_position_solution = sp.nsolve(
        (
            T_b_wrist[0, 3] - pw_end[0],
            T_b_wrist[1, 3] - pw_end[1],
            T_b_wrist[2, 3] - pw_end[2],
        ),
        (sp.symbols("theta1"), sp.symbols("theta2"), sp.symbols("theta3")),
        (0.1, 0.1, 0.1),  # Adjust the initial guess
        dict=True,
        bounds=[
            (-sp.pi / 2, sp.pi / 2),
            (-sp.pi / 2, sp.pi / 2),
            (-sp.pi / 2, sp.pi / 2),
        ],
    )
    T_b_wrist.evalf(
        subs={
            sp.symbols("theta1"): wrist_position_solution[0][sp.symbols("theta1")],
            sp.symbols("theta2"): wrist_position_solution[0][sp.symbols("theta2")],
            sp.symbols("theta3"): wrist_position_solution[0][sp.symbols("theta3")],
        }
    )
    # 2nd step wrist orientation
    R_end = sp.Matrix([[*n_end], [*s_end], [*a_end]]).T
    # calculating joint angles for the wrist
    R_0_3 = Robot.A[0][:3, :3] * Robot.A[1][:3, :3] * Robot.A[2][:3, :3]
    R_0_3_num = R_0_3.evalf(
        subs={
            sp.symbols("theta1"): wrist_position_solution[0][sp.symbols("theta1")],
            sp.symbols("theta2"): wrist_position_solution[0][sp.symbols("theta2")],
            sp.symbols("theta3"): wrist_position_solution[0][sp.symbols("theta3")],
        }
    )

    R_3_6_num = R_0_3_num.T * R_end

    R_3_6 = Robot.A[3][:3, :3] * Robot.A[4][:3, :3] * Robot.A[5][:3, :3]

    orientation_solution = sp.nsolve(
        (
            R_3_6[0, 2] - R_3_6_num[0, 2],
            R_3_6[1, 2] - R_3_6_num[1, 2],
            R_3_6[2, 2] - R_3_6_num[2, 2],
            R_3_6[0, 1] - R_3_6_num[0, 1],
            R_3_6[2, 0] - R_3_6_num[2, 0],
            R_3_6[1, 0] - R_3_6_num[1, 0],
        ),
        (sp.symbols("theta4"), sp.symbols("theta5"), sp.symbols("theta6")),
        (0.1, 0.1, 0.1),  # Adjust the initial guess
        dict=True,
        solver="bisect",
        bounds=[
            (-sp.pi / 2, sp.pi / 2),
            (-sp.pi / 2, sp.pi / 2),
            (-sp.pi / 2, sp.pi / 2),
        ],
    )

    full_solution = {**wrist_position_solution[0], **orientation_solution[0]}
    return full_solution


# %% Test the function for the intermediary position
h_agv = 0.3
p_end_effector_start = sp.Matrix([1.5, 0.5, h_agv + 0.1])
n_end_effector_start = sp.Matrix([1, 0, 0])
s_end_effector_start = sp.Matrix([0, -1, 0])
a_end_effector_start = sp.Matrix([0, 0, -1])

start_position_joints = reverse_kinematics(
    d_6,
    Robot,
    p_end_effector_start,
    n_end_effector_start,
    s_end_effector_start,
    a_end_effector_start,
)

threshold = 1e-10

start_position_joints = {
    k: 0 if abs(v) < threshold else v for k, v in start_position_joints.items()
}
start_position_joints

# %% Test the function for the end position
Robot.T_all.evalf(
    subs={
        sp.symbols("theta1"): start_position_joints[sp.symbols("theta1")],
        sp.symbols("theta2"): start_position_joints[sp.symbols("theta2")],
        sp.symbols("theta3"): start_position_joints[sp.symbols("theta3")],
        sp.symbols("theta4"): start_position_joints[sp.symbols("theta4")],
        sp.symbols("theta5"): start_position_joints[sp.symbols("theta5")],
        sp.symbols("theta6"): start_position_joints[sp.symbols("theta6")],
    }
)

# %% Linear joint functions of t (time)
for key in start_position_joints.keys():
    print(
        f"{key} = {start_position_joints[key]} + ({-start_position_joints[key]+end_position_joints[key]})t"
    )

# %% Add intermediary position
p_end_effector_intermediary = sp.Matrix([1.5, 0.2, 0.7])
n_end_effector_intermediary = sp.Matrix([1, 0, 0])
s_end_effector_intermediary = sp.Matrix([0, -1, 0])
a_end_effector_intermediary = sp.Matrix([0, 0, -1])

intermediary_position_joints = reverse_kinematics(
    d_6,
    Robot,
    p_end_effector_intermediary,
    n_end_effector_intermediary,
    s_end_effector_intermediary,
    a_end_effector_intermediary,
)

# %% Test the function for the intermediary position
intermediary_position_joints = {
    k: 0 if abs(v) < threshold else v for k, v in intermediary_position_joints.items()
}
intermediary_position_joints

Robot.T_all.evalf(
    subs={
        sp.symbols("theta1"): intermediary_position_joints[sp.symbols("theta1")],
        sp.symbols("theta2"): intermediary_position_joints[sp.symbols("theta2")],
        sp.symbols("theta3"): intermediary_position_joints[sp.symbols("theta3")],
        sp.symbols("theta4"): intermediary_position_joints[sp.symbols("theta4")],
        sp.symbols("theta5"): intermediary_position_joints[sp.symbols("theta5")],
        sp.symbols("theta6"): intermediary_position_joints[sp.symbols("theta6")],
    }
)

# %% Quadratic joint functions of t (time)
# find a quadratic function for each joint
# t = 0 -> start_position_joints
# t = 0.5 -> intermediary_position_joints
# t = 1 -> end_position_joints
for key in start_position_joints.keys():
    print(
        f"{key} = {start_position_joints[key]} + ({-2*start_position_joints[key]+4*intermediary_position_joints[key]-2*end_position_joints[key]})t^2"
    )