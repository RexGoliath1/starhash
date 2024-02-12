import numpy as np

# Function to rotate vectors by a given rotation matrix
def rotate_vectors(vectors, rotation_matrix):
    return np.dot(vectors, rotation_matrix.T)

def rotation_angle(rotation_matrix1, rotation_matrix2):
    rotation_matrix1 = normalize_rotation_matrix(rotation_matrix1)
    rotation_matrix2 = normalize_rotation_matrix(rotation_matrix2)

    # Compute the trace of the difference matrix
    trace_diff = np.trace(rotation_matrix1.T @ rotation_matrix2)
    trace_diff = np.clip(trace_diff, -1, 3)
    angle = np.arccos((trace_diff - 1) / 2)

    return np.degrees(angle)

def rotation_vector_from_matrices(rotation_matrix1, rotation_matrix2):
    R = rotation_matrix2 @ np.linalg.inv(rotation_matrix1)
    
    # Ensure the rotation matrix is orthogonal
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt

    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))

    if theta != 0:
        rx = (R[2, 1] - R[1, 2]) / (2 * np.sin(theta))
        ry = (R[0, 2] - R[2, 0]) / (2 * np.sin(theta))
        rz = (R[1, 0] - R[0, 1]) / (2 * np.sin(theta))
        axis = np.array([rx, ry, rz])
    else:
        axis = np.array([0, 0, 1])

    rotation_vector = axis * theta
    return rotation_vector


def normalize_rotation_matrix(rotation_matrix):
    # Orthogonalize the rotation matrix using QR decomposition
    Q, R = np.linalg.qr(rotation_matrix)
    # Ensure the determinant is positive
    if np.linalg.det(Q) < 0:
        Q *= -1
    return Q

def euler_to_rotation_matrix(yaw, pitch, roll):
    """ Convert Euler angles (yaw, pitch, roll) to rotation matrix """
    # Convert angles to radians
    yaw = np.radians(yaw)
    pitch = np.radians(pitch)
    roll = np.radians(roll)

    # Compute sine and cosine values
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)

    # Compute rotation matrix
    rotation_matrix = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]
    ])

    return rotation_matrix

def quaternion_to_dcm(q):
    q0 = q[3]
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]

    C = np.zeros((3, 3))
    C[0, 0] = q0**2 + q1**2 - q2**2 - q3**2
    C[0, 1] = 2 * (q1 * q2 + q0 * q3)
    C[0, 2] = 2 * (q1 * q3 - q0 * q2)
    C[1, 0] = 2 * (q1 * q2 - q0 * q3)
    C[1, 1] = q0**2 - q1**2 + q2**2 - q3**2
    C[1, 2] = 2 * (q2 * q3 + q0 * q1)
    C[2, 0] = 2 * (q1 * q3 + q0 * q2)
    C[2, 1] = 2 * (q2 * q3 - q0 * q1)
    C[2, 2] = q0**2 - q1**2 - q2**2 + q3**2

    return C

def dcm_to_quaternion(C):
    trace = np.trace(C)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1)
        q = np.array([(C[2, 1] - C[1, 2]) * s,
                      (C[0, 2] - C[2, 0]) * s,
                      (C[1, 0] - C[0, 1]) * s,
                      0.25 / s])
    elif C[0, 0] > C[1, 1] and C[0, 0] > C[2, 2]:
        s = 2.0 * np.sqrt(1.0 + C[0, 0] - C[1, 1] - C[2, 2])
        q = np.array([0.25 * s,
                      (C[0, 1] + C[1, 0]) / s,
                      (C[0, 2] + C[2, 0]) / s,
                      (C[2, 1] - C[1, 2]) / s])
    elif C[1, 1] > C[2, 2]:
        s = 2.0 * np.sqrt(1.0 + C[1, 1] - C[0, 0] - C[2, 2])
        q = np.array([(C[0, 1] + C[1, 0]) / s,
                      0.25 * s,
                      (C[1, 2] + C[2, 1]) / s,
                      (C[0, 2] - C[2, 0]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + C[2, 2] - C[0, 0] - C[1, 1])
        q = np.array([(C[0, 2] + C[2, 0]) / s,
                      (C[1, 2] + C[2, 1]) / s,
                      0.25 * s,
                      (C[1, 0] - C[0, 1]) / s])
    return q

