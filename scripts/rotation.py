import numpy as np
from tqdm import tqdm
from time import time

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


# Function to add noise to vectors
def add_noise(vectors, noise_level):
    noise = noise_level * np.random.randn(*vectors.shape)
    noisy_vectors = vectors + noise
    # Renormalize in case noise affects the magnitude
    norms = np.linalg.norm(noisy_vectors, axis=1, keepdims=True)
    return noisy_vectors / norms

def add_angle_noise(v, angle_noise_magnitude):
    angle_noise = (np.random.rand(3) - 0.5) * 2 * angle_noise_magnitude
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_noise[0]), -np.sin(angle_noise[0])],
                   [0, np.sin(angle_noise[0]), np.cos(angle_noise[0])]])
    
    Ry = np.array([[np.cos(angle_noise[1]), 0, np.sin(angle_noise[1])],
                   [0, 1, 0],
                   [-np.sin(angle_noise[1]), 0, np.cos(angle_noise[1])]])
    
    Rz = np.array([[np.cos(angle_noise[2]), -np.sin(angle_noise[2]), 0],
                   [np.sin(angle_noise[2]), np.cos(angle_noise[2]), 0],
                   [0, 0, 1]])
    
    R_noise = Rx @ Ry @ Rz
    v_noisy = R_noise @ v
    return v_noisy


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
    kappa = np.trace(delta * np.linalg.inv(S))

    # Coefficients for the characteristic equation
    a = sigma**2 - kappa
    b = sigma**2 + Z.T @ Z
    c = delta + Z.T @ S @ Z
    d = Z.T @ S @ S @ Z 
    constant = a*b + c*sigma - d

    # Newton-Raphson method to find the root
    lambda_ = np.sum(w)
    last_lambda = 0.0
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


def single_sample(vectors, angle, angle_noise_magnitude):
    """ Generate a single sample of QUEST Dispersions """
    vi = generate_random_unit_vectors(vectors)  # Inertial vectors

    # Define a small rotation (for simplicity, using a rotation around the z-axis)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle),  np.cos(angle), 0],
                                [0,              0,             1]])

    # Rotate vectors to simulate body vectors
    vb = rotate_vectors(vi, rotation_matrix)

    # Apply a small amount of noise to each vector
    # noise_level = 0.01 # Adjust as needed
    # vb_noisy = add_noise(vb, noise_level)
    vb_noisy = vb
    for ii in range(vectors):
        vb_noisy[ii] = add_angle_noise(vb[ii], angle_noise_magnitude)

    w = np.ones(vectors) * .25
    vi = vi.T
    vb_noisy = vb_noisy.T

    t1 = time()
    C_opt, q_opt = quest1981(vb_noisy, vi, w)
    dt = time() - t1
    angle_error = rotation_angle(rotation_matrix, C_opt)
    return angle_error, dt

def pipeline():
    fov = 25.0
    width = 720
    vectors = 5  # 4 vectors (4 minimum)
    angle = np.radians(10)  # 10 degree rotation
    angle_noise_magnitude = np.radians(0.01) # 0.1 degree errors
    samples = 10000

    angle_errors = np.zeros(samples)
    dt = np.zeros(samples)
    for ii in tqdm(range(samples)):
        angle_errors[ii], dt[ii] = single_sample(vectors, angle, angle_noise_magnitude)

    mu_err = np.mean(angle_errors)
    std_err = np.std(angle_errors)

    std_sim_pixels= width * (np.degrees(angle_noise_magnitude) / fov)
    mu_pixels = width * (mu_err / fov)
    std_pixels = width * (std_err / fov)

    print("___________________________________________________________________")
    print(f"Samples: {samples}. Vectors: {vectors}")
    print(f"Nominal z-axis rotation of {np.degrees(angle)} degrees")
    print(f"Simulated Star Vector Angle error of {np.degrees(angle_noise_magnitude)} degrees ({np.degrees(angle_noise_magnitude) * 3600} arcsec)")

    print("___________________________________________________________________")
    # Calculate the angle of rotation between rotation_matrix and C_opt
    print(f"Mean Attitude Error: {mu_err:.4f} degrees ({mu_err * 3600:.4f} arcsec)")
    print(f"Std. Dev. Attitude Error: {std_err:.4f} degrees ({std_err * 3600:.4f} arcsec)")

    print("___________________________________________________________________")
    print(f"FOV: {fov} degrees, Width: {width} pixels")
    print(f"Std. Dev. Induced Star Centroid Error: {std_sim_pixels:.3f} pixels")
    # Note: These aren't actually centroid errors, but just overall att errors in pixels
    print(f"Mean Attitude Error: {mu_pixels:.2f} pixels")
    print(f"Std. Dev. Attitude Error: {std_pixels:.2f} pixels")
    print("___________________________________________________________________")

    mu_dt = np.mean(dt)
    std_dt = np.mean(dt)
    print(f"QUEST Runtime. Mean: {mu_dt:.5f} sec., Std. Dev. {std_dt:.5f} sec.")

if __name__ == "__main__":
    # TODO: Ongoing Constraints Analysis 
    #   SEA/EEA/FOV
    #   Centroiding Accuracy
    #   FPS / Algo Latency / Image Latency
    #   Computation O(n)
    #   Catalog Density
    #   Catalog Memory
    #   Optical / Distortion Errors
    #   FPA Misalignment
    #   Blur / Vel / Accel
    pipeline()
