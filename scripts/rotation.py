import numpy as np

# Function to generate random unit vectors
def generate_random_unit_vectors(n):
    # Random vectors in 3D space
    vectors = np.random.rand(n, 3) - 0.5  # Shift to center around 0
    # Normalize to make them unit vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit_vectors = vectors / norms
    return unit_vectors

# Function to rotate vectors by a given rotation matrix
def rotate_vectors(vectors, rotation_matrix):
    return np.dot(vectors, rotation_matrix.T)

# Function to add noise to vectors
def add_noise(vectors, noise_level):
    noise = noise_level * np.random.randn(*vectors.shape)
    noisy_vectors = vectors + noise
    # Renormalize in case noise affects the magnitude
    norms = np.linalg.norm(noisy_vectors, axis=1, keepdims=True)
    return noisy_vectors / norms

def quaternion_to_dcm(q):
    """Convert a quaternion to a direction cosine matrix (DCM)."""
    q0, q1, q2, q3 = q
    C = np.array([
        [q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), q0**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), q0**2 - q1**2 - q2**2 + q3**2]
    ])
    return C

def quest1981(v_b, v_i, w):
    """Quaternion Estimator (QUEST) algorithm."""
    tolerance = 1e-5

    # B matrix computation
    B = v_b @ np.diag(w) @ v_i.T
    S = B + B.T
    sigma = np.trace(B)

    # Helper variables
    Z = np.array([B[1, 2] - B[2, 1], B[2, 0] - B[0, 2], B[0, 1] - B[1, 0]])
    delta = np.linalg.det(S)
    kappa = np.trace(np.linalg.inv(S) @ S @ S)

    # Coefficients for the characteristic equation
    a = sigma**2 - kappa
    b = sigma**2 + np.dot(Z, Z)
    c = delta + np.dot(Z, S @ Z)
    d = np.dot(Z, S @ S @ Z)
    constant = a*b + c*sigma - d

    # Newton-Raphson method to find the root
    lambda_ = np.sum(w)
    last_lambda = 0
    while np.abs(lambda_ - last_lambda) >= tolerance:
        last_lambda = lambda_
        f = lambda_**4 - (a + b)*lambda_**2 - c*lambda_ + constant
        f_dot = 4*lambda_**3 - 2*(a + b)*lambda_ - c
        lambda_ -= f / f_dot

    # Optimal quaternion computation
    omega = lambda_
    alpha = omega**2 - sigma**2 + kappa
    beta = omega - sigma
    gamma = (omega + sigma)*alpha - delta
    X = (alpha*np.eye(3) + beta*S + S @ S) @ Z
    q_opt = np.hstack((X, gamma)) / np.sqrt(gamma**2 + np.linalg.norm(X)**2)

    # Convert quaternion to DCM
    C_opt = quaternion_to_dcm(q_opt)

    return C_opt, q_opt

# Generate 4 random unit vectors for "star" locations
vi = generate_random_unit_vectors(4)  # Inertial vectors

# Define a small rotation (for simplicity, using a rotation around the z-axis)
angle = np.radians(10)  # 10 degree rotation
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                            [np.sin(angle),  np.cos(angle), 0],
                            [0,              0,             1]])

# Rotate vectors to simulate body vectors
vb = rotate_vectors(vi, rotation_matrix)

# Apply a small amount of noise to each vector
noise_level = 0.00000001  # Adjust as needed
vb_noisy = add_noise(vb, noise_level)

w = np.ones(4)
vi = vi.T
vb_noisy = vb_noisy.T
print(f"vi.shape: {vi.shape}")
print(f"vb_noisy.shape: {vb_noisy.shape}")
print(f"w.shape: {w.shape}")

C_opt, q_opt = quest1981(vb_noisy, vi, w)

print(f"rotation_matrix: {rotation_matrix}")
print(f"C_opt: {C_opt}")
print("Done")

# Function to calculate the angle of rotation between two rotation matrices
def rotation_angle(rotation_matrix1, rotation_matrix2):
    # Compute the trace of the difference matrix
    trace_diff = np.trace(rotation_matrix1.T @ rotation_matrix2)
    # Clamp the trace to the valid range [-1, 3] to avoid numerical errors
    trace_diff = np.clip(trace_diff, -1, 3)
    # Calculate the angle of rotation (in radians)
    angle = np.arccos((trace_diff - 1) / 2)
    # Convert radians to degrees
    angle_degrees = np.degrees(angle)
    return angle_degrees

# Calculate the angle of rotation between rotation_matrix and C_opt
angle_of_rotation = rotation_angle(rotation_matrix.T, C_opt)
print("Angle of rotation between rotation_matrix and C_opt:", angle_of_rotation, "degrees")
