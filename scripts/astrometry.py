from astroquery.astrometry_net import AstrometryNet
from astropy.wcs import WCS
from glob import glob
import numpy as np
import os
import astropy.units as u
import cv2
import exifread
from astropy.io import fits
from photutils.aperture import CircularAperture
from photutils.centroids import centroid_com
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from time import time

import warnings
warnings.filterwarnings("ignore", category=Warning)

ast = AstrometryNet()
ast.api_key = "ikuwhswwrpdfmezb"
submission_id = None
try_again = True


# Vizier inputs
search_radius = 15.0 * u.deg

ra_key = None
dec_key = None
force_image_upload = False
ra_dec_units = None

# Color wavelength range to estimate pixel scale
# Taken from max zoom settings of Sony A6400 + SEL E 18-135mm F/3.5-5.6 OSS Lens
f_number = 5.6
sensor_width = 23.5 * 10**-3
sensor_height = 15.6 * 10**-3
focal_length = 135.0 * 10**-3
aperture = focal_length / f_number
min_wave_length = 300.0 * 10**-9
max_wave_length = 800.0 * 10**-9

# Approximation based on the Rayleigh criterion for circular apertures: https://en.wikipedia.org/wiki/Angular_resolution
min_scale = 1.22 * min_wave_length / (aperture) * 180.0 / np.pi * 3600.0
max_scale = 1.22 * max_wave_length / (aperture) * 180.0 / np.pi * 3600.0

settings = {
    "scale_units": "arcsecperpix",
    "scale_type": "ul",
    "scale_lower": min_scale,
    "scale_upper": max_scale,
    "fwhm": 15.0,
    "detect_threshold": 30.0,
    "solve_timeout": 360.0,
    "crpix_center": True,
    "force_image_upload": False, 
    "parity": 0,
    "publicly_visible": "n",
}


# Current file path, go down one level to get to the image folder
image_folder = os.path.join(os.path.dirname(__file__), '../images')
fits_folder = os.path.join(image_folder + '/fits')
os.makedirs(fits_folder, exist_ok=True)

# Find all jpg files in the folder using glob
image_list = glob(image_folder + '/*.jpeg')

# First, convert jpg images to fits
for image_path in image_list:
    #tags = exifread.process_file(open(image_path, 'rb'))
    #image_width = float(tags['EXIF ExifImageWidth'].values[0])
    #image_height = float(tags['EXIF ExifImageLength'].values[0])
    #focal_length = float(tags['EXIF FocalLength'].values[0])
    #f_number = float(tags['EXIF FNumber'].values[0])
    #aperture = focal_length / f_number

    fits_path = os.path.join(fits_folder, os.path.basename(image_path).split('.')[0] + '.fits')
    if os.path.exists(fits_path):
        continue

    # Load JPEG image and convert to numpy array
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Create FITS HDU and write to file
    hdu = fits.PrimaryHDU(image)
    hdu.writeto(fits_path, overwrite=True)

# Find all fits files in the folder using glob
#image_list = glob(fits_folder + '/*.fits')
image_list = glob(image_folder + '/*.jpeg')

# Upload all images to astrometry.net
for image in image_list:
    #image_data = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    #image_shape = image_data.shape
    try_again = True
    while try_again:
        print(f'Uploading {image}')
        try:
            if not submission_id:
                t1 = time()
                #wcs_header = ast.solve_from_image(image, submission_id=submission_id, crpix_center=True, solve_timeout=120.0, force_image_upload=True)
                #wcs_header = ast.solve_from_image(image, submission_id=submission_id, **settings)
                #wcs_header = ast.solve_from_image(image, submission_id=submission_id, crpix_center=True, solve_timeout=360.0, force_image_upload=True)
                wcs_header = ast.solve_from_image(image, submission_id=submission_id, solve_timeout=600.0, force_image_upload=True)
                t2 = time()
                print("Solving took {} seconds".format(t2 - t1))
            else:
                wcs_header = ast.monitor_submission(submission_id, solve_timeout=timeout)
        except TimeoutError as e:
            submission_id = e.args[1]
        else:
            try_again = False

            #pixel_scale_x = np.abs(wcs_header['CD1_1'] * 3600.0) * u.arcsecond/u.pixel
            #pixel_scale_y = np.abs(wcs_header['CD2_2'] * 3600.0) * u.arcsecond/u.pixel
            #fov_x = image_width * pixel_scale_x.to(u.deg)
            #fov_y = image_height * pixel_scale_y.to(u.deg)
            #fov = max(fov_x, fov_y)

            if wcs_header:
                wcs = WCS(wcs_header)
                viz = Vizier(columns=['HIP', 'RA_ICRS', 'DE_ICRS'], row_limit=-1)
                catalog = viz.query_region( \
                    wcs.pixel_to_world(0, 0), \
                    radius=search_radius, \
                    catalog='I/311/hip_main' \
                )

                sky_coords = SkyCoord(ra=catalog[0]['RA_ICRS'], dec=catalog[0]['DE_ICRS'], unit=u.deg)

                # Convert the sky coordinates to pixel coordinates
                x_pixels, y_pixels = wcs.world_to_pixel(sky_coords)
                # Only keep stars within the image bounds
                mask = (x_pixels >= 0) & (x_pixels <= image_shape[1]) & (y_pixels >= 0) & (y_pixels <= image_shape[0])
                x_pixels, y_pixels = x_pixels[mask], y_pixels[mask]
                for x, y in zip(x_pixels, y_pixels):
                    print(f"Star located at pixel coordinates ({x:.2f}, {y:.2f})")

            else:
                print('nok')
