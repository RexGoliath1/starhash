import os
import numpy as np
from tqdm import tqdm
from time import time
import math
from scipy.stats import kurtosis
import matplotlib.pyplot as plt

outdir = os.path.join(os.path.dirname(__file__), "att_det_results")
os.makedirs(outdir, exist_ok=True)

methods = {
    1: "davenport",
    2: "quest",
    3: "triad"
}

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

def rotation_angle(rotation_matrix1, rotation_matrix2):
    rotation_matrix1 = normalize_rotation_matrix(rotation_matrix1)
    rotation_matrix2 = normalize_rotation_matrix(rotation_matrix2)

    # Compute the trace of the difference matrix
    trace_diff = np.trace(rotation_matrix1.T @ rotation_matrix2)
    trace_diff = np.clip(trace_diff, -1, 3)
    angle = np.arccos((trace_diff - 1) / 2)

    return np.degrees(angle)

def normalize_rotation_matrix(rotation_matrix):
    # Orthogonalize the rotation matrix using QR decomposition
    Q, R = np.linalg.qr(rotation_matrix)
    # Ensure the determinant is positive
    if np.linalg.det(Q) < 0:
        Q *= -1
    return Q

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

def quest(v_b, v_i, w):
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

def davenport(v_b, v_i, w):
    # Matrix K (4x4)
    #B = np.dot((v_b * w[:, np.newaxis]).T, v_i.T)
    B = v_b @ np.diag(w) @ v_i.T
    Z = np.array([B[1, 2] - B[2, 1], B[2, 0] - B[0, 2], B[0, 1] - B[1, 0]])
    Z = Z.reshape(3,1)
    K = np.block([[B + B.T - np.eye(3) * np.trace(B), Z],
                  [Z.T, np.trace(B)]])

    # Eigenvector associated with largest eigenvalue
    evals, evecs = np.linalg.eig(K)
    idx = np.argmax(evals)
    q_opt = evecs[:, idx]
    C_opt = quaternion_to_dcm(q_opt);

    return C_opt, q_opt

def triad(b1, b2, r1, r2):
    """Tri-axial Attitude Determination (TRIAD)"""

    # Calculate v2
    v2 = np.cross(r1, r2) / np.linalg.norm(np.cross(r1, r2))

    # Calculate w2
    w2 = np.cross(b1, b2) / np.linalg.norm(np.cross(b1, b2))

    # Calculate rotation matrix C
    C_opt = np.outer(b1, r1) + np.outer(np.cross(b1, w2), np.cross(r1, v2)) + np.outer(w2, v2)

    # Convert rotation matrix to quaternion
    q_opt = dcm_to_quaternion(C_opt)

    return C_opt, q_opt

def single_sample(method, vectors, rotation, angle_noise_magnitude):
    """ Generate a single sample of QUEST Dispersions """
    vi = generate_random_unit_vectors(vectors)  # Inertial vectors

    yaw, pitch, roll = rotation
    rotation_matrix = euler_to_rotation_matrix(yaw, pitch, roll)

    # Rotate vectors to simulate body vectors
    vb = rotate_vectors(vi, rotation_matrix)

    # Apply a small amount of noise to each vector
    # noise_level = 0.01 # Adjust as needed
    # vb_noisy = add_noise(vb, noise_level)
    vb_noisy = vb
    for ii in range(vectors):
        vb_noisy[ii] = add_angle_noise(vb[ii], angle_noise_magnitude)

    w = np.ones(vectors)
    vi = vi.T
    vb_noisy = vb_noisy.T

    t1 = time()
    if method == "davenport":
        C_opt, q_opt = davenport(vb_noisy, vi, w)
    elif method == "quest":
        C_opt, q_opt = quest(vb_noisy, vi, w)
    elif method == "triad":
        # Note: Assume first vector is known. Has error even when zero noise.
        # vb1 = vb[0, :]
        vb1 = vb_noisy[:, 0]
        vb2 = vb_noisy[:, 1]
        vi1 = vi[:, 0]
        vi2 = vi[:, 1]
        C_opt, q_opt = triad(vb1, vb2, vi1, vi2)
    else:
        raise Exception(f"Unknown method {method}")

    dt = time() - t1
    angle_error = rotation_angle(rotation_matrix, C_opt)
    return angle_error, dt

def print_errors(method, config, angle_errors, dt):
    stars = config["stars"]
    samples = config["samples"]
    vectors = config["vectors"]
    width = config["width"]
    fov = config["fov"]
    angle_noise_magnitude = config["angle_noise_magnitude"]

    pcat_size = int(math.factorial(stars) / (math.factorial(stars-vectors) * math.factorial(vectors)))


    max_err = np.max(angle_errors)
    min_err = np.min(angle_errors)
    mu_err = np.mean(angle_errors)
    med_err = np.median(angle_errors)
    std_err = np.std(angle_errors)
    kurt_err = kurtosis(angle_errors)

    std_sim_pixels = width * (np.degrees(angle_noise_magnitude) / fov)
    mu_pixels = width * (mu_err / fov)
    std_pixels = width * (std_err / fov)

    print("___________________________________________________________________")
    print(f"{method.upper()} METHOD")
    print("___________________________________________________________________")
    print(f"Samples: {samples}. Vectors: {vectors}")
    print(f"Simulated Star Vector Angle error of {np.degrees(angle_noise_magnitude)} degrees ({np.degrees(angle_noise_magnitude) * 3600} arcsec)")
    print(f"Catalog Size: {stars} stars, Pattern Catalog Size: {pcat_size}, Pattern Catalog Memory: {pcat_size * 5 / (1024**3):.2f} GB")


    print("___________________________________________________________________")
    print(f"Max Attitude Error: {max_err:.4f} degrees ({max_err * 3600:.4f} arcsec)")
    print(f"Min Attitude Error: {min_err:.4f} degrees ({min_err * 3600:.4f} arcsec)")
    print(f"Mean Attitude Error: {mu_err:.4f} degrees ({mu_err * 3600:.4f} arcsec)")
    print(f"Median Attitude Error: {med_err:.4f} degrees ({med_err * 3600:.4f} arcsec)")
    print(f"Std. Dev. Attitude Error: {std_err:.4f} degrees ({std_err * 3600:.4f} arcsec)")
    print(f"Attitude Error Kurtosis: {kurt_err:.4f}")

    print("___________________________________________________________________")
    print(f"FOV: {fov} degrees, Width: {width} pixels")
    print(f"Std. Dev. Induced Star Centroid Error: {std_sim_pixels:.3f} pixels")
    print(f"Mean Attitude Error: {mu_pixels:.2f} pixels")
    print(f"Std. Dev. Attitude Error: {std_pixels:.2f} pixels")
    print("___________________________________________________________________")

    mu_dt = np.mean(dt)
    std_dt = np.std(dt)
    print(f"Runtime. Mean: {mu_dt:.5f} sec., Std. Dev. {std_dt:.5f} sec.")

    h = plt.hist(x=angle_errors, bins=1000, edgecolor="k")
    plt.title(f"{method.upper()} Attitude Error Histograms")
    plt.vlines(mu_err, 0, np.max(h[0]), colors="r", linestyles='dashed', label='')
    plt.vlines(med_err, 0, np.max(h[0]), colors="b", linestyles='dashed', label='hello world?')
    plt.vlines(mu_err + std_err, 0, np.max(h[0]), colors="g", linestyles='dashed', label='')
    plt.vlines(mu_err - std_err, 0, np.max(h[0]), colors="g", linestyles='dashed', label='')

    plt.xlabel("Attitude Error (degrees)")

    plt.xlim([0, 0.1])
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{method.upper()} Attitude_Error"))
    plt.close()

    h = plt.hist(x=dt, bins=1000, edgecolor="k")
    plt.title(f"{method.upper()} Runtime Histograms")
    plt.xlabel("Runtime (seconds)")
    plt.xlim([0, 0.0001])
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{method.upper()} Runtime"))
    plt.close()

def pipeline():
    config = {
        "fov" : 25.0,
        "width" : 720,
        "vectors" : 4, # 4 vectors (4 minimum)
        "angle_noise_magnitude" : np.radians(0.05),
        "samples" : 10000,
        "stars" : 1000
    }

    samples = config["samples"]
    vectors = config["vectors"]
    angle_noise_magnitude = config["angle_noise_magnitude"]

    # Generate samples
    angle_errors = np.zeros(samples)
    dt = np.zeros(samples)
    rotations = np.radians(np.random.uniform(0, 360, [samples, 3]))

    for id in methods.keys():
        method = methods[id]
        for ii in tqdm(range(samples)):
            angle_errors[ii], dt[ii] = single_sample(method, vectors, rotations[ii], angle_noise_magnitude)
        print_errors(method, config, angle_errors, dt)


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
