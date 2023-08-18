import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

dir_path = os.path.dirname(os.path.realpath(__file__))
img_path = os.path.dirname(dir_path)
# test_img = os.path.join(img_path, "images/m35.jpg")
test_img = os.path.join(img_path, "images/generic_starfield.png")
assert (os.path.exists(test_img))
img = cv2.imread(test_img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
denoise = True
poi_threhsold = 8
poi_max_size = 50  # GIANT: 50
poi_min_size = 2  # GIANT: 2
reject_saturation = True
centroid_size = 2


def get_outliers(samples, sigma_cutoff=4):
    # compute the distance each sample is from the median of the samples
    median_distances = np.abs(np.array(samples) - np.median(samples))

    # the median of the median distances
    median_distance = np.median(median_distances)

    # compute the median distance sigma score for each point
    median_sigmas = 1.4826*median_distances / \
        median_distance if median_distance else median_distances / \
        np.mean(median_distances)

    # find outliers based on the specified sigma level
    outliers = median_sigmas >= sigma_cutoff

    return outliers


def gaussian_fit(x, y, z):
    x = np.array(x).ravel()
    y = np.array(y).ravel()
    z = np.array(z).ravel()

    # form the Jacobian matrix
    # we are fitting to a model of the form
    coefficients = np.vstack(
        [np.power(x, 2), x, np.power(y, 2), y, np.ones(x.shape)]).T

    try:
        solution = np.linalg.lstsq(coefficients, np.log(z), rcond=None)[0]

        sigma_x = np.sqrt(-1 / (2 * solution[0]))
        sigma_y = np.sqrt(-1 / (2 * solution[2]))

        x0 = solution[1] * sigma_x ** 2
        y0 = solution[3] * sigma_y ** 2

        amplitude = np.exp(
            solution[4] + x0 ** 2 / (2 * sigma_x ** 2) + y0 ** 2 / (2 * sigma_y ** 2))

        centroid_x = x0
        centroid_y = y0

        if (sigma_x < 0) or (sigma_y < 0):
            print("Sigmas are negative ...")
            return np.nan, np.nan

        computed = amplitude * np.exp(-(x - centroid_x) ** 2 / (
            2 * sigma_x ** 2) - (y - centroid_y) ** 2 / (2 * sigma_y ** 2))

        residuals = z - computed

        jacobian = np.vstack(
            [computed * (x - centroid_x) / (sigma_x ** 2),
             computed * (y - centroid_y) / (sigma_y ** 2),
             computed * (x - centroid_x) ** 2 / (sigma_x ** 3),
             computed * (y - centroid_y) ** 2 / (sigma_y ** 3),
             computed / amplitude]).T

        covariance = np.linalg.pinv(
            jacobian.T @ jacobian) * float(np.std(residuals)) ** 2

    except np.linalg.linalg.LinAlgError:
        print("Something bad happened")
        return np.nan, np.nan
    return centroid_x, centroid_y


if denoise:
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

flat_image = gray.astype(np.float32) - \
    cv2.medianBlur(gray.copy().astype(np.float32), 5)

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

data = (flat_image[next_rows, next_cols] -
        flat_image[start_rows, start_cols]).ravel()

outliers = get_outliers(data)
standard_deviation = np.nanstd(data[~outliers]) / 2

snr = flat_image / standard_deviation

interesting_pix = snr > poi_threhsold
_, __, stats, ___ = cv2.connectedComponentsWithStats(
    interesting_pix.astype(np.uint8))

poi_subs = []
out_stats = []
out_snrs = []


# loop through each grouping of pixels
for blob in stats:
    if (poi_max_size >= blob[-1]) and (blob[-1] >= poi_min_size):
        # get the subscript to the maximum illumination value within the current component and appen
        # return list
        poi_roi = flat_image[blob[1]:blob[1] +
                             blob[3], blob[0]:blob[0] + blob[2]]

        # get the x/y location by unraveling the index (and reversing the order
        local_subs = np.unravel_index(
            np.nanargmax(poi_roi), poi_roi.shape)[::-1]
        # store the results translated back to the full image and the statistics
        poi_subs.append(local_subs + blob[[0, 1]])
        out_stats.append(blob)
        out_snrs.append(snr[blob[1]:blob[1] + blob[3],
                        blob[0]:blob[0] + blob[2]].max())


# initialize lists for output
star_points = []
star_illums = []
star_psfs = []
out_stats = []

psf_img = img_rgb.copy()
color1 = (0, 255, 0)
color2 = (255, 0, 0)
radius1 = 5
radius2 = 10
thickness = 1
plot1 = True
plot2 = True


# loop through the pixel level points of interest
for ind, center in enumerate(poi_subs):
    column_array = np.arange(
        center[0] - centroid_size, center[0] + centroid_size + 1)
    row_array = np.arange(
        center[1] - centroid_size, center[1] + centroid_size + 1)
    col_check = (column_array >= 0) & (column_array <= gray.shape[1] - 1)
    row_check = (row_array >= 0) & (row_array <= gray.shape[0] - 1)
    # valid_check = col_check & row_check
    cols, rows = np.meshgrid(column_array[col_check], row_array[row_check])

    if plot1:
        psf_img = cv2.circle(psf_img, (int(center[0]), int(center[1])),
                             radius1, color1, thickness)

    # if col_check and row_check:
    if cols.size >= 0.5 * (2 * centroid_size + 1)**2:
        sampled_image = gray[rows, cols].astype(np.float64)
        x0, y0 = gaussian_fit(cols, rows, sampled_image)
        # x0, y0 = psf.centroid

        if (x0 < 0) or (y0 < 0) or (np.isnan((x0, y0)).any()):
            print(f"No fit: {ind}")
            continue
        else:
            if plot2:
                psf_img = cv2.circle(psf_img, (int(x0), int(y0)),
                                     radius2, color2, thickness)

        if (np.abs(np.asarray(center) - np.asarray([x0, y0]).flatten()) <= 3).all():
            star_points.append([x0, y0])
            star_illums.append(gray[tuple(center[::-1])])
            # star_psfs.append(psf)
            if stats is not None:
                out_stats.append(stats[ind])

plt.figure(figsize=(40, 15))
plt.imshow(psf_img)
# plt.imshow(snr.astype(np.uint8), 'gray')
plt.show()

print(f"Read the image {test_img}")
