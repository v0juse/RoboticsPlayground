import numpy as np


def rotation_matrix(axis, theta, radians=True):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    if not radians:
        theta = np.deg2rad(theta)
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def Rx(gamma, radians=True):
    if not radians:
        gamma = np.deg2rad(gamma)
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)],
        ]
    )


def Ry(beta, radians=True):
    if not radians:
        beta = np.deg2rad(beta)
    return np.array(
        [
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)],
        ]
    )


def Rz(alpha, radians=True):
    if not radians:
        alpha = np.deg2rad(alpha)
    return np.array(
        [
            [np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1],
        ]
    )


def rotate_vector(vector, alpha, beta, gamma, radians=True):
    """
    Rotate a vector by alpha (z axis rotation), beta: (y axis rotation),
    and gamma (z axis rotation).
    """
    if not radians:
        alpha = np.deg2rad(alpha)
        beta = np.deg2rad(beta)
        gamma = np.deg2rad(gamma)
    Rz = rotation_matrix([0, 0, 1], alpha)
    Ry = rotation_matrix([0, 1, 0], beta)
    Rx = rotation_matrix([1, 0, 0], gamma)
    R = np.dot(Rz, np.dot(Ry, Rx))
    return np.dot(R, vector)


# Problem 1
vector = [0, 0, 1]
gamma = 10
beta = 30
alpha = 45
print("Rotated vector 1: ", rotate_vector(vector, alpha, beta, gamma, radians=False))

# Problem 2
## a) fixed axis - ZYX RzRyRx
gamma = alpha = beta = 90
Rzyx = np.dot(
    Rz(alpha, radians=False), np.dot(Ry(beta, radians=False), Rx(gamma, radians=False))
)
print("Rzyx: ", Rzyx)
print("Rotated vector 2a: ", np.dot(Rzyx, vector))

## b) intrinsic axis - XYZ RxRyRz
Rxyz = np.dot(Rx(gamma), np.dot(Ry(beta), Rz(alpha)))
print("Rxyz", Rxyz)
print("Rotated vector 2b", np.dot(Rxyz, vector))