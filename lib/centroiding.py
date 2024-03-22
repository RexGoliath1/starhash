import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from time import time
from tqdm import tqdm
import argparse
import yaml
from glob import iglob
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

# TODO: Proper Module
import sys
from pathlib import Path
lib_path = Path(__file__).resolve().parent.parent / 'lib'
sys.path.append(str(lib_path))
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from transform_utils import rotation_vector_from_matrices, vectors_to_angle
from attitude_determination import quest, davenport, triad, foma, svd, esoq2
from stellar_utils import StellarUtils

# Default Paths
CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
REPO_DIRECTORY = os.path.dirname(CURRENT_DIRECTORY)
IMAGE_DIRECTORY = os.path.join(REPO_DIRECTORY, "stellarscript", "results")
OUTPUT_DIRECTORY = os.path.join(CURRENT_DIRECTORY, "results")
DEFAULT_CONFIG = os.path.join(REPO_DIRECTORY, "data", "centroiding.yaml")

blob_kwargs = {
    "color": (0, 255, 0),
    "radius": 5,
    "thickness": 1,
}

gauss_kwargs = {
    "color": (255, 0, 0),
    "radius": 10,
    "thickness": 1,
}

truth_kwargs = {
    "color": (0, 0, 255),
    "radius": 20,
    "thickness": 1,
}

def group_poi_stats(flat_image, snr, stats, args):
    poi_subs = []
    poi_stats = []
    poi_snrs = []

    # loop through each grouping of pixels
    for blob in stats:
        if (args.poi_max_size >= blob[-1]) and (blob[-1] >= args.poi_min_size):
            poi_roi = flat_image[blob[1]:blob[1] + blob[3], blob[0]:blob[0] + blob[2]]

            # get the x/y location by unraveling the index (and reversing the order
            local_subs = np.unravel_index(np.nanargmax(poi_roi), poi_roi.shape)[::-1]
            # store the results translated back to the full image and the statistics
            poi_subs.append(local_subs + blob[[0, 1]])
            poi_stats.append(blob)
            poi_snrs.append(snr[blob[1]:blob[1] + blob[3], blob[0]:blob[0] + blob[2]].max())

    return poi_subs, poi_stats, poi_snrs

def process_params(args):
    # Override parameters provided in the config
    if args.config_path is not None and os.path.exists(args.config_path):
        with open(args.config_path, 'r') as stream:
            config = yaml.safe_load(stream)
            for param in config.keys():
                setattr(args, param, config[param])

    if args.output_plots:
        os.makedirs(args.output_path, exist_ok=True)
    else:
        args.display_blob_centroids = False
        args.display_gauss_centroids = False
        args.output_individual_blobs = False

    assert (os.path.exists(args.input_path))

def get_closest_points(truth, measured, pix_thresh=10.0):
    """ Use KDTree/Hungarian to map centroids to truth """
    max_cost = pix_thresh * 1000
    assert(pix_thresh < max_cost)
    tree = cKDTree(truth)
    idx = tree.query_ball_point(x=measured, r=pix_thresh, p=2)
    cost_matrix = np.full((len(measured), len(truth)), max_cost)

    for i, indices in enumerate(idx):
        for truth_index in indices:
            distance = np.linalg.norm(np.array(truth[truth_index]) - np.array(measured[i]))
            cost_matrix[i, truth_index] = distance

    measured_indicies, truth_indices = linear_sum_assignment(cost_matrix)

    valid_assignments = cost_matrix[measured_indicies, truth_indices] < pix_thresh
    measured_indicies = measured_indicies[valid_assignments]
    truth_indices = truth_indices[valid_assignments]
    return measured_indicies, truth_indices


def plot_centroid(fig_name, ind, dx, dy, roi_snr, roi, z0, residuals, covariance):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Blob {ind} Centroiding", fontsize=20)

    gray_plt = axs[0, 0].imshow(roi, cmap='gray')
    axs[0, 0].scatter(dx, dy, marker='x', color='red', s=100)  # Scatter plot of centroid as red "x"
    axs[0, 0].set_title(f"Centroid: {dx:.2f} {dy:.2f}")
    plt.colorbar(gray_plt, ax=axs[0, 0], fraction=0.046, pad=0.04)

    max_snr = np.max(roi_snr)
    snr_plt = axs[0, 1].imshow(roi_snr, cmap='Reds', interpolation='nearest', vmin=0.0, vmax=max_snr)
    axs[0, 1].set_title(f"SNR (thresh: {args.snr_threshold})")
    plt.colorbar(snr_plt, ax=axs[0, 1], fraction=0.046, pad=0.04)

    # max_z = np.max(z0)
    # z_plt = axs[1, 0].imshow(z0.reshape(roi.shape), cmap='Reds', interpolation='nearest', vmax=max_z)
    # axs[1, 0].set_title(f"Least Squares")
    # plt.colorbar(z_plt, ax=axs[1, 0], fraction=0.046, pad=0.04)

    max_z = np.max(covariance)
    min_z = np.max(covariance)
    z_plt = axs[1, 0].imshow(covariance, cmap='Reds', interpolation='nearest', vmax=max_z, vmin=min_z)
    axs[1, 0].set_title(f"Covariance")
    plt.colorbar(z_plt, ax=axs[1, 0], fraction=0.046, pad=0.04)

    max_res = np.max(residuals)
    res_plt = axs[1, 1].imshow(residuals.reshape(roi.shape), cmap='Reds', interpolation='nearest', vmax=max_res)
    axs[1, 1].set_title(f"Residuals")
    plt.colorbar(res_plt, ax=axs[1, 1], fraction=0.046, pad=0.04)

    plt.draw()
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.clf()
    plt.close(fig)

def centroiding_pipeline(image_path, args, stel):
    """ Centroiding Loop """
    centroid_params = {}
    centroid_size = args.centroid_size
    distance_threshold = args.distance_threshold

    process_params(args)
    img = cv2.imread(image_path)

    np.random.seed(args.seed)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    centroid_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if args.output_plots:
        os.makedirs(args.image_output_path, exist_ok=True)
        print(f"TRUTH : {stel.truth_coords.shape}")
        for ii in range(stel.truth_coords.shape[1]):
            [x,y] = stel.truth_coords[:, ii]
            centroid_img = cv2.circle(centroid_img, (int(x), int(y)), **truth_kwargs)

    if args.denoise:
        gray = cv2.GaussianBlur(gray, (args.size_denoise, args.size_denoise), 0)

    print(f"gray.shape: {gray.shape}")
    flat_image = gray.astype(np.float32) - cv2.medianBlur(gray.copy(), args.size_median)

    im_shape = flat_image.shape
    dist = np.minimum(np.min(im_shape) - 1, distance_threshold)
    num_pix = float(np.prod(np.array(im_shape) - dist))
    num_choice = int(np.minimum(num_pix // 4, 2000))

    # Randomized pixel choise for starting row / col. May need more thought
    rand_idx = np.random.choice(np.arange(int(num_pix)), num_choice, replace=False)
    start_rows, start_cols = np.unravel_index(rand_idx, np.array(im_shape) - dist)

    next_rows = start_rows + dist
    next_cols = start_cols + dist

    diff_flat_image = (flat_image[next_rows, next_cols] - flat_image[start_rows, start_cols]).ravel()
    outliers = get_outliers(diff_flat_image)

    standard_deviation = np.nanstd(diff_flat_image[~outliers]) / 2
    snr = flat_image / standard_deviation
    snr_thresh_img = snr > args.snr_threshold
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(snr_thresh_img.astype(np.uint8))
    poi_subs, poi_stats, poi_snr = group_poi_stats(flat_image, snr, stats, args)

    # initialize lists for output
    star_centroids = []
    star_illums = []
    out_stats = []
    gauss_illums = []

    blob_candidates = len(poi_subs)
    blob_candidate = np.zeros((blob_candidates, 1))

    # loop through the pixel level points of interest
    for ind, center in enumerate(tqdm(poi_subs)):
        column_array = np.arange(center[0] - centroid_size, center[0] + centroid_size + 1)

        row_array = np.arange(center[1] - centroid_size, center[1] + centroid_size + 1)

        col_check = (column_array >= 0) & (column_array <= gray.shape[1] - 1)
        row_check = (row_array >= 0) & (row_array <= gray.shape[0] - 1)
        cols, rows = np.meshgrid(column_array[col_check], row_array[row_check])

        if args.display_blob_centroids:
            centroid_img = cv2.circle(centroid_img, (int(center[0]), int(center[1])), **blob_kwargs)

        # Edge check for the current centroid roi (too small for estimate)
        if cols.size < 0.5 * (2 * centroid_size + 1)**2:
            continue

        roi = gray[rows, cols]
        roi = roi.astype(np.float64)
        roi_snr = snr[rows, cols]
        x0, y0, z0, residuals, covariance = gaussian_fit(cols, rows, roi)

        if args.display_gauss_centroids:
            if not ((x0 < 0) or (y0 < 0) or (np.isnan((x0, y0)).any())):
                # centroid_img = cv2.circle(centroid_img, (int(x0), int(y0)), **gauss_kwargs)
                if (np.abs(np.asarray(center) - np.asarray([x0, y0]).flatten()) <= 3).all():
                    centroid_img = cv2.circle(centroid_img, (int(x0), int(y0)), **gauss_kwargs)

        if not ((x0 < 0) or (y0 < 0) or (np.isnan((x0, y0)).any())):
            blob_candidate[ind, 0] = 1
            dx = x0 - (center[0] - centroid_size)
            dy = y0 - (center[1] - centroid_size)
            if (np.abs(np.asarray(center) - np.asarray([x0, y0]).flatten()) <= 3).all():
                star_centroids.append([x0, y0])
                star_illums.append(gray[tuple(center[::-1])])
                out_stats.append(stats[ind])
                gauss_illums.append(np.min(z0, 0).sum())


        if args.output_individual_blobs:
            fig_name = os.path.join(args.image_output_path, f"blob_{ind}.png")
            if (x0 < 0) or (y0 < 0) or (np.isnan((x0, y0)).any()):
                plt.imshow(roi, 'gray')
                plt.title(f"No fit: Blob {ind}: {x0}, {y0}")
                plt.draw()
                plt.savefig(fig_name)
                plt.clf()
                continue
            else:
                plot_centroid(fig_name, ind, dx, dy, roi_snr, roi, z0, residuals, covariance)

    pct = 100.0 * blob_candidate.sum() / blob_candidates
    print(f"Centroid candidates passing Gaussian Fit: {blob_candidate.sum()} / {blob_candidates} ({pct} %)")

    magnitude_idx = np.argsort(gauss_illums, kind="mergesort")[::-1]
    star_centroids = np.array(star_centroids)[magnitude_idx]
    star_illums = np.array(star_illums)[magnitude_idx]
    gauss_illums = np.array(gauss_illums)[magnitude_idx]
    out_stats = np.array(out_stats)[magnitude_idx]

    centroid_params["star_centroids"] = star_centroids
    centroid_params["star_illums"] = star_illums
    centroid_params["gauss_illums"] = gauss_illums
    centroid_params["out_stats"] = out_stats

    if args.output_plots:
        meas_idx, truth_idx = get_closest_points(stel.truth_coords_list, star_centroids)

        pct = 100.0 * blob_candidate.sum() / blob_candidates
        plt.figure(figsize=(40, 15))
        plt.imshow(centroid_img)
        plt.title(f"Full Solution Blob Gaussian Percent: {pct:2f}")
        plt.draw()
        fig_name = os.path.join(args.image_output_path, f"full_solution.png")
        plt.savefig(fig_name)

    return centroid_params


def get_outliers(samples, sigma_cutoff=4):
    # compute the distance each sample is from the median of the samples
    median_distances = np.abs(np.array(samples) - np.median(samples))

    # the median of the median distances
    median_distance = np.median(median_distances)

    # compute the median distance sigma score for each point
    median_sigmas = 1.4826 * median_distances / \
        median_distance if median_distance else median_distances / \
        np.mean(median_distances)

    # find outliers based on the specified sigma level
    outliers = median_sigmas >= sigma_cutoff

    return outliers

def gaussian_fit(x, y, z):
    x = np.array(x).ravel()
    y = np.array(y).ravel()
    z = np.array(z).ravel()
    z += 1

    # form the Jacobian matrix we are fitting to a model of the form
    coefficients = np.vstack(
        [np.power(x, 2), x, np.power(y, 2), y, np.ones(x.shape)]).T

    #try:
    
    epsilon = 1e-10
    z_safe = z.clip(min=epsilon)
    solution = np.linalg.lstsq(coefficients, np.log(z_safe), rcond=1e-15)[0]

    if solution[0] > 0 or solution[2] > 0:
        print("TODO: Solutions are positive ...")
        return np.nan, np.nan, np.nan, np.nan, np.nan

    sigma_x = np.sqrt(-1 / (2 * solution[0]))
    sigma_y = np.sqrt(-1 / (2 * solution[2]))

    x0 = solution[1] * sigma_x ** 2
    y0 = solution[3] * sigma_y ** 2

    amplitude = np.exp(
        solution[4] + x0 ** 2 / (2 * sigma_x ** 2) + y0 ** 2 / (2 * sigma_y ** 2))

    if (sigma_x < 0) or (sigma_y < 0):
        print("Sigmas are negative ...")
        return np.nan, np.nan, np.nan, np.nan, np.nan

    z0 = amplitude * np.exp(-(x - x0) ** 2 / (2 * sigma_x ** 2) - (y - y0) ** 2 / (2 * sigma_y ** 2))
    residuals = z - z0

    jacobian = np.vstack(
        [z0 * (x - x0) / (sigma_x ** 2),
         z0 * (y - y0) / (sigma_y ** 2),
         z0 * (x - x0) ** 2 / (sigma_x ** 3),
         z0 * (y - y0) ** 2 / (sigma_y ** 3),
         z0 / amplitude]).T

    covariance = np.linalg.pinv(
        jacobian.T @ jacobian) * float(np.std(residuals)) ** 2
    #except np.linalg.LinAlgError as e:
    #    print(f"Probably couldn't invert: {e}")
    #    return np.nan, np.nan, np.nan, np.nan, np.nan
    #except ValueError as e:
    #    print(f"Something bad happened: {e}")
    #    return np.nan, np.nan, np.nan, np.nan, np.nan
        #except:
        #    print("what in the fuck")
        #    return np.nan, np.nan, np.nan, np.nan, np.nan

    return x0, y0, z0, residuals, covariance

def run_pipeline(args, stel):

    process_params(args)

    for image_path in iglob(os.path.join(stel.image_path, args.image_pattern)):
        image_name = os.path.splitext(os.path.basename(image_path))
        args.image_output_path = os.path.join(args.output_path, image_name[0])

        if not stel.get_stel_data(image_path):
            continue

        # Block for repeatedly running centroiding for runtime performance checks
        if args.test_runtime:
            num_test = 1
            test = np.zeros(num_test)

            for ii in range(num_test):
                t1 = time()
                _ = centroiding_pipeline(image_path, args, stel)
                test[ii] = time() - t1
            print(f"Took {np.mean(test):.2f} seconds on average. Std: {np.std(test):.2f}. {num_test} samples")
        else:
            centroid_params = centroiding_pipeline(image_path, args, stel)
            star_centroids = centroid_params["star_centroids"]
            star_centroids = np.array(star_centroids)
            num_centroids = len(star_centroids)
            if num_centroids == 0:
                print(f"No centroids found for {image_path}")
                continue

            measured_coords = np.zeros([2, num_centroids])
            measured_vectors = np.zeros([3, num_centroids])

            for jj, star_centroid in enumerate(star_centroids):
                [u,v] = star_centroid
                pc = np.array([[u, v, 1]],).T
                vc = stel.K_inv @ pc
                vc[2] = np.sqrt(1 - vc[0]**2 -vc[1]**2)
                measured_vectors[:, jj] = vc.T
                measured_coords[:, jj] = star_centroid

            meas_idx, truth_idx = get_closest_points(stel.truth_coords_list, star_centroids)

            # If successful, vb and vi should be the 1:1 mapping of centroid vectors to inertial truth
            vb = measured_vectors[:, meas_idx]
            mc = measured_coords[:, meas_idx]
            vi = stel.truth_vectors[:, truth_idx]
            tc = stel.truth_coords[:, truth_idx]
            vc_truth = stel.truth_body_vectors[:, truth_idx]

            # Sanity checks
            for vnum in range(0, len(stel.truth_vectors)):
                print(f"Truth: ({tc[:, vnum]}) ; Measured: ({mc[:, vnum]})")
                print(f"Truth Vector : ({vc_truth[:, vnum]}) ; Measured Vector: ({vb[:, vnum]})")
                ang = np.degrees(vectors_to_angle(vc_truth[:, vnum], vb[:, vnum]))
                print(f"Angle Between Centroid and Star: {ang:.2f}")

            # TODO: Plot all attitude accuracy combinations of pattern size
            quest_vectors = 4
            if quest_vectors is not None and quest_vectors < vb.shape[1]:
                rng = np.random.default_rng()
                vectors = rng.choice(vb.shape[1], size=quest_vectors, replace=False)
                vb = vb[:, vectors]
                vi = vi[:, vectors]

            C_opt, q_opt = quest(vb, vi)
            rot_vec_error = rotation_vector_from_matrices(stel.T_cam_to_j2000, C_opt)
            rv_deg = np.degrees(rot_vec_error)
            rv_as = rv_deg * 3600
            rv_mas = 1000 * rv_as

            print(f"rv_mag_mas: {np.linalg.norm(rv_mas):.2f} rot_vec_error: {rv_mas} (Milli Arc-Seconds)")
            print(f"rv_mag_as: {np.linalg.norm(rv_as):.2f} rot_vec_error: {rv_as} (Arc-Seconds)")
            print(f"rv_mag_deg: {np.linalg.norm(rv_deg):.2f} rot_vec_error: {rv_deg} (Degrees)")

            print("Finished matching!")

if __name__ == "__main__":
    """ Simple Star Centroiding """
    print("Starting....")
    parser = argparse.ArgumentParser(description="Overlay OpenCV blob detection on an image")
    parser.add_argument("--input_path", default=IMAGE_DIRECTORY, type=str, help="")
    parser.add_argument("--output_path", default=OUTPUT_DIRECTORY ,type=str, help="")
    parser.add_argument("--config_path", default=DEFAULT_CONFIG,type=str, help="")
    args = parser.parse_args()

    stel = StellarUtils(args.input_path)
    run_pipeline(args, stel)
