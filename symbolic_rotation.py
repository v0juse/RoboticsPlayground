# %% Importing libraries
import sympy as sp
import pydantic as pyd
import numpy as np


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

    def set_joint_functions(self, joint_functions: list[sp.Function]):
        self.joint_functions = joint_functions
        # NOTE is it the best way to do this implicitly?
        self.J = self.get_jacobian_matrix()

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

    def get_jacobian_matrix(self):
        assert hasattr(self, "joint_functions"), "Joint functions must be set"

        q = self.joint_functions.keys()
        J = sp.zeros(6, len(q))

        T = sp.eye(4)
        z = sp.Matrix([0, 0, 1])
        p = sp.Matrix([0, 0, 0])

        for i in range(len(q)):
            T = T * self.A[i]
            z_i = T[0:3, 2]
            p_i = T[0:3, 3]

            J_v = z.cross(p_i - p)
            J_w = z

            J[:, i] = J_v.col_join(J_w)
            z = z_i
            p = p_i

        return sp.simplify(J)

    def calculate_joint_velocities(self, t: sp.Symbol | float):
        q_dot = [fn.diff(t) for fn in self.joint_functions.values()]
        joint_velocities = self.J * sp.Matrix(q_dot)
        linear_velocities = joint_velocities[:3, :]
        angular_velocities = joint_velocities[3:, :]
        return linear_velocities, angular_velocities

    def calculate_end_effector_velocities(self):
        q_dot = [fn.diff(t) for fn in self.joint_functions.values()]
        return self.J * sp.Matrix(q_dot)

    def calculate_inverse_velocity_kinematics(self, end_effector_velocities):
        J_inv = self.J.pinv()
        q_dot = J_inv * sp.Matrix(end_effector_velocities)
        return q_dot


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
p_end_final = sp.Matrix([1.5, -0.1, 0.7])
n_end_final = sp.Matrix([sp.sqrt(2) / 2, sp.sqrt(2) / 2, 0])
s_end_final = sp.Matrix([sp.sqrt(2) / 2, -sp.sqrt(2) / 2, 0])
a_end_final = sp.Matrix([0, 0, -1])


pw_end = p_end_final - d_6 * a_end_final
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
R_end = sp.Matrix([[*n_end_final], [*s_end_final], [*a_end_final]]).T
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
def inverse_kinematics(
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

start_position_joints = inverse_kinematics(
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
# input: (ti, qi) and (tf, qf)
# find a linear function for each joint
# TODO find equations
# [ ] be printable
# [ ] plot graphs in time
joint_functions = {}
t = sp.symbols("t")
for key in start_position_joints.keys():
    fn = (
        start_position_joints[key]
        + (-start_position_joints[key] + end_position_joints[key]) * t
    )
    joint_functions[key] = fn
    joint_functions[key] = sp.simplify(joint_functions[key])
    print(f"{key} = {joint_functions[key]}")


# %% Add intermediary position
p_end_effector_intermediary = sp.Matrix([1.5, 0.2, 0.9])
n_end_effector_intermediary = sp.Matrix([sp.sqrt(2) / 2, sp.sqrt(2) / 2, 0])
s_end_effector_intermediary = sp.Matrix([sp.sqrt(2) / 2, -sp.sqrt(2) / 2, 0])
a_end_effector_intermediary = sp.Matrix([0, 0, -1])

intermediary_position_joints = inverse_kinematics(
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

# %% Print the tree sets of joint functions
print("Start position joints:")
for key, value in start_position_joints.items():
    print(f"\t{key} = {value}")
print("Intermediary position joints:")
for key, value in intermediary_position_joints.items():
    print(f"\t{key} = {value}")
print("End position joints:")
for key, value in end_position_joints.items():
    print(f"\t{key} = {value}")


# %% Linear joint functions of t (time)
# find a quadratic function for each joint
# t = 0 -> start_position_joints
# t = 0.5 -> intermediary_position_joints
# t = 1 -> end_position_joints
def linear_function(t, a, b):
    return a + (b - a) * t


joint_functions = {}
n = 0
for key in start_position_joints.keys():
    qi = start_position_joints[key]
    qm = intermediary_position_joints[key]
    qf = end_position_joints[key]
    fn = sp.Piecewise(
        (linear_function(t, qi, qm), t < 1), (linear_function((t - 1), qm, qf), t >= 1)
    )
    joint_functions[key] = fn

    n += 1
    print(f"θ_{{{n}}} = {joint_functions[key]}")
    # print(
    #     f"θ_{{{n}}} = If(t<1,{linear_function(t, qi, qm)},{linear_function((t- 1), qm, qf)})"
    # )
# %% test
joint_functions[sp.symbols("theta1")].subs(t, 0.49)
joint_functions[sp.symbols("theta1")].subs(t, 0.5)

#  %% Plotting the joint functions of t (time)

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Define the time range for plotting
time_range = np.linspace(0, 2, 100)

# Plotting the joint functions
plt.figure(figsize=(10, 6))


# Formatter function to convert y-axis values to fractions of π
def pi_fraction_formatter(x, pos):
    frac = sp.Rational(x / np.pi).limit_denominator()
    if frac == 0:
        return "0"
    elif frac == 1:
        return "$\pi$"
    elif frac == -1:
        return "$-\pi$"
    elif frac.numerator == 1:
        return f"$\pi/{frac.denominator}$"
    elif frac.numerator == -1:
        return f"$-\pi/{frac.denominator}$"
    else:
        return f"${frac.numerator}\pi/{frac.denominator}$"


# Predefine specific y-tick values
yticks = np.array(
    [
        -np.pi / 2,
        -np.pi / 3,
        -np.pi / 4,
        -np.pi / 5,
        -np.pi / 7,
        -np.pi / 10,
        0,
        np.pi / 10,
        np.pi / 7,
        np.pi / 5,
        np.pi / 4,
        np.pi / 3,
        np.pi / 2,
    ]
)

# Plotting the joint functions
plt.figure(figsize=(10, 7), dpi=500)

for key, fn in joint_functions.items():
    # Create a numpy function from the sympy expression
    fn_np = sp.lambdify(t, fn, "numpy")
    # Evaluate the function over the time range
    fn_values = fn_np(time_range)
    # Plot the function
    plt.plot(time_range, fn_values, label=key)

# Add labels and legend
plt.xlabel("Tempo Normalizado (t)", fontsize=14)
plt.ylabel("Ângulos de Junta (rad)", fontsize=14)
plt.title("Funções de Junta Lineares", fontsize=16)
plt.legend(fontsize=12, loc="upper left")
plt.grid(True)

# Set the y-axis ticks and labels
plt.gca().set_yticks(yticks)
plt.gca().yaxis.set_major_formatter(FuncFormatter(pi_fraction_formatter))

# Increase the size of the x and y ticks
plt.tick_params(axis="both", which="major", labelsize=12)

# Show the plot
plt.show()

# %% Discontinues joint velocity
Robot.set_joint_functions(joint_functions)
discontinuous_linear_velocity, discontinuous_angular_speed = (
    Robot.calculate_joint_velocities(t)
)
discontinuous_linear_velocity, discontinuous_angular_speed
Robot.calculate_end_effector_velocities()
# %% Plot velocities


# time_range = np.linspace(0, 2, 100)

# for key, fn in discontinuous_linear_velocity.items():
#    # Create a numpy function from the sympy expression
#     fn_np = sp.lambdify(t, fn, "numpy")
#     # Evaluate the function over the time range
#     fn_values = fn_np(time_range)
#     # Plot the function
#     plt.plot(time_range, fn_values, label=key)

# %% Optimal (3th degree whatever) joint functions of t (time)
# Conditions:
# 1. t=0, y=a
# 2. t=1, y=b
# 3. t=0, del y = 0
# 4. t=1, del y = 0
# NOTE: more equations
# a3t^3 + a2t^2 + a1t + a0 = qf
# 3a3t^2 + 2a2t + a1 = wf

import sympy as sp


def third_degree_interpolation(t, qi, qf, w0, wf):
    # Define symbols
    a0 = qi
    a1 = w0
    a2 = sp.symbols("a2")
    a3 = sp.symbols("a3")

    # Solve for a2 and a3
    solved_values = sp.nsolve(
        (
            a3 * 1**3 + a2 * 1**2 + a1 * 1 + a0 - qf,
            3 * a3 * 1**2 + 2 * a2 * 1 + a1 - wf,
        ),
        (a2, a3),
        (1, 1),  # Initial guess for a2 and a3
    )

    # Extract solved values for a2 and a3
    a2_solved, a3_solved = solved_values

    # Use solved values in the return statement
    return a0 + a1 * t + a2_solved * t**2 + a3_solved * t**3


t = sp.symbols("t")
optimal_joint_functions = {}

# find Optimal function for each joint
# t = 0 -> start_position_joints
# t = 1 -> intermediary_position_joints
# t = 2 -> end_position_joints
n = 1
for key in start_position_joints.keys():
    qi = start_position_joints[key]
    qm = intermediary_position_joints[key]
    qf = end_position_joints[key]
    wi = 0
    wm = 0.5
    wf = 0

    first_half_fn = third_degree_interpolation(t, qi, qm, wi, wm)
    second_half_fn = third_degree_interpolation(t - 1, qm, qf, wm, wf)

    fn = sp.Piecewise((first_half_fn, t < 1), (second_half_fn, t >= 1))

    print(f"θ_{{{n}}} = If(t<1,{first_half_fn},{second_half_fn})")
    n += 1
    optimal_joint_functions[key] = sp.simplify(fn)
optimal_joint_functions
# %% plot optimal joint functions

# Define the time range for plotting
time_range = np.linspace(0, 2, 100)

# Plotting the joint functions
plt.figure(figsize=(10, 6))


# Plotting the joint functions
plt.figure(figsize=(10, 7), dpi=500)

for key, fn in optimal_joint_functions.items():
    # Create a numpy function from the sympy expression
    fn_np = sp.lambdify(t, fn, "numpy")
    # Evaluate the function over the time range
    fn_values = fn_np(time_range)
    # Plot the function
    plt.plot(time_range, fn_values, label=key)

# Add labels and legend
plt.xlabel("Tempo Normalizado (t)", fontsize=14)
plt.ylabel("Ângulos de Junta (rad)", fontsize=14)
plt.title("Funções de Junta Optimas", fontsize=16)
plt.legend(fontsize=12, loc="upper left")
plt.grid(True)

# Set the y-axis ticks and labels
plt.gca().set_yticks(yticks)
plt.gca().yaxis.set_major_formatter(FuncFormatter(pi_fraction_formatter))

# Increase the size of the x and y ticks
plt.tick_params(axis="both", which="major", labelsize=12)

# Show the plot
plt.show()

# %% Optimal joint velocity
Robot.set_joint_functions(optimal_joint_functions)
optimal_linear_velocity, optimal_angular_speed = Robot.calculate_joint_velocities(t)
optimal_linear_velocity, optimal_angular_speed
end_effector_velocities = Robot.calculate_end_effector_velocities()
end_effector_velocities
# # %%plot velocities
# linear_velocities = end_effector_velocities[:3, :]
# angular_velocities = end_effector_velocities[3:, :]

# # linear_velocities = np.array(
# #     [np.array(v).astype(np.float64).flatten() for v in linear_velocities]
# # )
# # angular_velocities = np.array(
# #     [np.array(v).astype(np.float64).flatten() for v in angular_velocities]
# # )

# fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# time_points = np.linspace(0, 2, 100)

# # Plot linear velocities
# axs[0].plot(time_points, linear_velocities[:, 0], label="vx")
# axs[0].plot(time_points, linear_velocities[:, 1], label="vy")
# axs[0].plot(time_points, linear_velocities[:, 2], label="vz")
# axs[0].set_title("Linear Velocities")
# axs[0].set_xlabel("Time [s]")
# axs[0].set_ylabel("Velocity [units/s]")
# axs[0].legend()
# axs[0].grid()

# # Plot angular velocities
# axs[1].plot(time_points, angular_velocities[:, 0], label="wx")
# axs[1].plot(time_points, angular_velocities[:, 1], label="wy")
# axs[1].plot(time_points, angular_velocities[:, 2], label="wz")
# axs[1].set_title("Angular Velocities")
# axs[1].set_xlabel("Time [s]")
# axs[1].set_ylabel("Velocity [rad/s]")
# axs[1].legend()
# axs[1].grid()

# plt.tight_layout()
# plt.show()

# %% Inverse Reverse Kinematics
# Calculate the Jacobian matrix
joint_symbols = list(start_position_joints.keys())
Robot.set_joint_functions({symbol: sp.Function(str(symbol))(sp.Symbol("t")) for symbol in joint_symbols})
Jacobian_matrix = Robot.J
Jacobian_matrix

# %% Define desired end-effector velocities
desired_linear_velocity = sp.Matrix([0.1, 0.2, 0.3])  # Example linear velocities
desired_angular_velocity = sp.Matrix([0.01, 0.02, 0.03])  # Example angular velocities
desired_velocity = desired_linear_velocity.col_join(desired_angular_velocity)

# %% Calculate the inverse Jacobian and joint velocities
Jacobian_inverse = Jacobian_matrix.evalf(subs=start_position_joints).pinv()
joint_velocities = Jacobian_inverse * desired_velocity
joint_velocities 
# %% Evaluate the joint velocities for a given robot configuration
joint_values = {
    sp.symbols("theta1"): sp.pi / 4,
    sp.symbols("theta2"): -sp.pi / 4,
    sp.symbols("theta3"): sp.pi / 6,
    sp.symbols("theta4"): -sp.pi / 6,
    sp.symbols("theta5"): sp.pi / 3,
    sp.symbols("theta6"): -sp.pi / 3,
}

joint_velocities_evaluated = joint_velocities.evalf(subs=joint_values)
joint_velocities_evaluated
# %%
