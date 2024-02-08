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

# Default Paths
CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
REPO_DIRECTORY = os.path.dirname(CURRENT_DIRECTORY)
IMAGE_DIRECTORY = os.path.join(REPO_DIRECTORY, "images")
OUTPUT_DIRECTORY = os.path.join(CURRENT_DIRECTORY, "results")
DEFAULT_CONFIG = os.path.join(CURRENT_DIRECTORY, "centroiding.yaml")

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

def parameter_override(args):
    # Override parameters provided in the config
    if args.config_path and os.path.exists(args.config_path):
        with open(args.config_path, 'r') as stream:
            config = yaml.safe_load(stream)

            for param in config.keys():
                if param in config and hasattr(args, param):
                    setattr(args, param, config[param])
    return args


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

def centroiding_pipeline(image_path, args):
    """ Centroiding Loop """
    np.random.seed(args.seed)
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if args.output_plots:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_name = os.path.splitext(os.path.basename(image_path))
        output_path = os.path.join(args.output_path, image_name[0])
        os.makedirs(output_path, exist_ok=True)

    if args.denoise:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

    flat_image = gray.astype(np.float32) - cv2.medianBlur(gray.copy().astype(np.float32), 5)

    im_shape = flat_image.shape
    dist = np.minimum(np.min(im_shape) - 1, 5)
    num_pix = float(np.prod(np.array(im_shape) - dist))

    num_choice = int(np.minimum(num_pix // 4, 2000))

    num_pix = float(np.prod(np.array(im_shape) - dist))

    start_rows, start_cols = np.unravel_index(
        np.random.choice(np.arange(int(num_pix)),
                         num_choice,
                         replace=False),
        np.array(im_shape) - dist)

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
    star_points = []
    star_illums = []
    star_psfs = []
    out_stats = []

    if args.output_plots and args.display_blob_centroids:
        psf_img = img_rgb.copy()

    blob_candidates = len(poi_subs)
    blob_candidate = np.zeros((blob_candidates, 1))

    # loop through the pixel level points of interest
    for ind, center in enumerate(tqdm(poi_subs)):
        column_array = np.arange(center[0] - args.centroid_size, center[0] + args.centroid_size + 1)

        row_array = np.arange(
            center[1] - args.centroid_size, center[1] + args.centroid_size + 1)

        col_check = (column_array >= 0) & (column_array <= gray.shape[1] - 1)
        row_check = (row_array >= 0) & (row_array <= gray.shape[0] - 1)
        cols, rows = np.meshgrid(column_array[col_check], row_array[row_check])

        if args.output_plots and args.display_blob_centroids:
            psf_img = cv2.circle(psf_img, (int(center[0]), int(center[1])), **blob_kwargs)

        # if col_check and row_check:
        if cols.size >= 0.5 * (2 * args.centroid_size + 1)**2:
            roi = gray[rows, cols]
            roi = roi.astype(np.float64)
            roi_snr = snr[rows, cols]
            x0, y0, z0, residuals, covariance = gaussian_fit(cols, rows, roi)
            # x0, y0 = psf.centroid
            if not ((x0 < 0) or (y0 < 0) or (np.isnan((x0, y0)).any())):
                blob_candidate[ind, 0] = 1
                dx = x0 - (center[0] - args.centroid_size)
                dy = y0 - (center[1] - args.centroid_size)
                if (np.abs(np.asarray(center) - np.asarray([x0, y0]).flatten()) <= 3).all():
                    star_points.append([x0, y0])
                    star_illums.append(gray[tuple(center[::-1])])
                    # star_psfs.append(psf)
                    if stats is not None:
                        out_stats.append(stats[ind])

            if args.output_plots:
                fig_name = os.path.join(output_path, f"blob_{ind}.png")
                if (x0 < 0) or (y0 < 0) or (np.isnan((x0, y0)).any()):
                    plt.imshow(roi, 'gray')
                    plt.title(f"No fit: Blob {ind}: {x0}, {y0}")
                    plt.draw()
                    plt.savefig(fig_name)
                    plt.clf()
                    continue
                else:
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

                    max_z = np.max(z0)
                    z_plt = axs[1, 0].imshow(z0.reshape(roi.shape), cmap='Reds', interpolation='nearest', vmax=max_z)
                    axs[1, 0].set_title(f"Least Squares")
                    plt.colorbar(z_plt, ax=axs[1, 0], fraction=0.046, pad=0.04)

                    max_res = np.max(residuals)
                    res_plt = axs[1, 1].imshow(residuals.reshape(roi.shape), cmap='Reds', interpolation='nearest', vmax=max_res)
                    axs[1, 1].set_title(f"Residuals")
                    plt.colorbar(res_plt, ax=axs[1, 1], fraction=0.046, pad=0.04)

                    plt.draw()
                    plt.tight_layout()
                    #plt.show()
                    plt.savefig(fig_name)
                    plt.clf()
                    plt.close(fig)
                    if args.display_gauss_centroids:
                        psf_img = cv2.circle(psf_img, (int(x0), int(y0)), **gauss_kwargs)

    pct = 100.0 * blob_candidate.sum() / blob_candidates
    print(f"Got {blob_candidate.sum()} / {blob_candidates} ({pct} %)")

    if args.output_plots:
        pct = 100.0 * blob_candidate.sum() / blob_candidates
        return np.nan, np.nan, np.nan, np.nan, np.nan
        plt.figure(figsize=(40, 15))
        plt.imshow(psf_img)
        plt.title(f"Full Solution Blob Gaussian Percent: {pct:2f}")
        #plt.show()
        plt.draw()
        fig_name = os.path.join(output_path, f"full_solution.png")
        plt.savefig(fig_name)


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

    sigma_x = np.sqrt(-1 / (2 * solution[0]))
    sigma_y = np.sqrt(-1 / (2 * solution[2]))

    x0 = solution[1] * sigma_x ** 2
    y0 = solution[3] * sigma_y ** 2

    amplitude = np.exp(
        solution[4] + x0 ** 2 / (2 * sigma_x ** 2) + y0 ** 2 / (2 * sigma_y ** 2))

    if (sigma_x < 0) or (sigma_y < 0):
        print("Sigmas are negative ...")
        return np.nan, np.nan

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

if __name__ == "__main__":
    """ Simple Star Centroiding """
    print("Starting....")
    parser = argparse.ArgumentParser(description="Overlay OpenCV blob detection on an image")
    parser.add_argument("--input_path", default=IMAGE_DIRECTORY, type=str, help="")
    parser.add_argument("--input_pattern", default="stellarium*.png", type=str, help="")
    parser.add_argument("--output_path", default=OUTPUT_DIRECTORY ,type=str, help="")
    parser.add_argument("--config_path", default=DEFAULT_CONFIG,type=str, help="")
    parser.add_argument("--output_plots", type=bool, default=False, help="")
    parser.add_argument("--display_blob_centroids", type=bool, default=True, help="")
    parser.add_argument("--display_gauss_centroids", type=bool, default=True, help="")
    parser.add_argument("--denoise", type=bool, default=True, help="")
    parser.add_argument("--flatten", type=bool, default=True, help="")
    parser.add_argument("--num_blobs", type=int, default=5, help="")
    parser.add_argument("--snr_threshold", type=int, default=5, help="")
    parser.add_argument("--poi_max_size", type=int, default=50, help="")
    parser.add_argument("--poi_min_size", type=int, default=2, help="")
    parser.add_argument("--reject_saturation", type=bool, default=True, help="")
    parser.add_argument("--centroid_size", type=int, default=True, help="")
    parser.add_argument("--seed", type=int, default=1337, help="")

    args = parser.parse_args()
    args = parameter_override(args)
    os.makedirs(args.output_path, exist_ok=True)
    
    assert (os.path.exists(args.input_path))
    assert (os.path.exists(args.output_path))

    #temp_path = "/Users/stevengonciar/git/starhash/stellarscript/results/20240206-071258/images"
    #image_path = "/Users/stevengonciar/git/starhash/stellarscript/results/20240206-071258/images"
    # Read in the K Matrix
    for image_path in iglob(os.path.join(args.input_path, "*8115.png")):
        num_test = 100
        test = np.zeros(num_test)
        print(test.shape)
        for ii in range(num_test):
            t1 = time()
            centroiding_pipeline(image_path, args)
            test[ii] = time() - t1
    print(f"Took {np.mean(test):.2f} seconds on average. Std: {np.std(test):.2f}. {num_test} samples")
