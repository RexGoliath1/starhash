import os
import numpy as np
from tqdm import tqdm
from time import time
import math
from scipy.stats import kurtosis
import matplotlib.pyplot as plt

# TODO: Proper Module
import sys
from pathlib import Path
lib_path = Path(__file__).resolve().parent.parent / 'lib'
sys.path.append(str(lib_path))

from transform_utils import rotate_vectors, rotation_angle, euler_to_rotation_matrix
from attitude_determination import quest, davenport, triad, foma, svd, esoq2

outdir = os.path.join(os.path.dirname(__file__), "att_det_results")
os.makedirs(outdir, exist_ok=True)

methods = {
    #1: "davenport",
    2: "quest",
    3: "triad",
    4: "foma",
    5: "svd",
    6: "esoq2"
}

# Function to generate random unit vectors
def generate_random_unit_vectors(n):
    # Random vectors in 3D space
    vectors = np.random.rand(n, 3) - 0.5  # Shift to center around 0
    # Normalize to make them unit vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit_vectors = vectors / norms
    return unit_vectors

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
        vb1 = vb_noisy[:, 0]
        vb2 = vb_noisy[:, 1]
        vi1 = vi[:, 0]
        vi2 = vi[:, 1]
        C_opt, q_opt = triad(vb1, vb2, vi1, vi2)
    elif method == "foma":
        C_opt, q_opt = foma(vb_noisy, vi, w)
    elif method == "svd":
        C_opt, q_opt = svd(vb_noisy, vi, w)
    elif method == "esoq2":
        C_opt, q_opt = esoq2(vb_noisy, vi, w)
    else:
        raise Exception(f"Unknown method {method}")

    dt = time() - t1
    angle_error = rotation_angle(rotation_matrix, C_opt)
    # assert(np.sum(C_opt @ vb_noisy - vi) < 0.1)
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

    plt.xlim([0, 0.2])
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"Attitude_Error - {method.upper()}"))
    plt.close()

    h = plt.hist(x=dt, bins=1000, edgecolor="k")
    plt.title(f"{method.upper()} Runtime Histograms")
    plt.xlabel("Runtime (seconds)")
    plt.xlim([0, 0.0001])
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"Runtime - {method.upper()}"))
    plt.close()

def pipeline():
    config = {
        "fov" : 25.0,
        "width" : 720,
        "vectors" : 4, # 4 vectors (4 minimum)
        "angle_noise_magnitude" : np.radians(0.0),
        # "angle_noise_magnitude" : np.radians(0.1),
        "samples" : 100000,
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
