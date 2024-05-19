from giant.camera import Camera
from giant.camera_models import PinholeModel
from giant.camera_models import BrownModel
from giant.point_spread_functions import Gaussian
from scipy.spatial.transform import Rotation as SciRot
from giant.ray_tracer.scene import Scene
from giant.image import OpNavImage
from giant.rotations import Rotation
from giant.stellar_opnav.star_identification import StarID
from giant.catalogues.giant_catalogue import GIANTCatalogue
import subprocess
import os
import shutil

import time
import numpy as np
from copy import copy
from datetime import datetime
import cv2
from tqdm import tqdm
import pickle

# Local Modules
from image_processing import centroid

def rot_mat_2(ra, dec):
    r_j2000_to_bore = SciRot.from_euler('ZYX', [-ra, dec, 0.0], degrees=False)
    r_bore_to_camera = SciRot.from_euler('zyx', [90.0, 0.0, 90.0], degrees=True)
    r =  r_bore_to_camera * r_j2000_to_bore
    return r.as_matrix()


def rotation_matrix(ra, dec):
    # Convert degrees to radians
    ra = np.radians(ra)
    dec = np.radians(dec)
    # Rotation matrix for RA around the z-axis
    Rz = np.array([
        [np.cos(ra), -np.sin(ra), 0],
        [np.sin(ra), np.cos(ra), 0],
        [0, 0, 1]
    ])
    # Rotation matrix for Dec around the y-axis
    Ry = np.array([
        [np.cos(dec), 0, np.sin(dec)],
        [0, 1, 0],
        [-np.sin(dec), 0, np.cos(dec)]
    ])
    # Combined rotation matrix
    return np.dot(Ry, Rz) # Rotate for Dec first then RA

def create_gif_ffmpeg(frame_pattern, output_gif, framerate=10):
    # Generate palette
    subprocess.run([
        "ffmpeg", "-framerate", str(framerate), "-i", frame_pattern,
        "-filter_complex", "palettegen", "-y", "palette.png"
    ])
    # Create GIF
    subprocess.run([
        "ffmpeg", "-framerate", str(framerate), "-i", frame_pattern,
        "-i", "palette.png", "-filter_complex", "paletteuse", "-y", output_gif
    ])

width = 1024
height = 1024
width = 1944
height = 2592

# fov = 25
# ar = width / height
# focal_length = (width / 2) / np.tan(np.radians(fov / 2))
# kx = focal_length
# ky = focal_length * (height / width)

# focal_length = np.tan(np.radians(fov/2)) 
# kx = (width / 2) / np.tan(np.radians(fov/2))
# ky = (height / 2) / np.tan(np.radians(fov/2))
# px = width / 2.0
# py = height / 2.0


# MODEL = PinholeModel(kx=kx, ky=ky, px=px, py=py, focal_length=focal_length, n_rows=height, n_cols=width)

MODEL = BrownModel(fx=3470, fy=3470, k1=-0.5, k2=0.3, k3=-0.2, p1=2e-5, p2=8e-5, px=1260, py=950,
                   n_rows=1944, n_cols=2592, field_of_view=25)  # type: BrownModel
# PSF = Gaussian(sigma_x=1, sigma_y=1, size=5)
PSF = Gaussian(sigma_x=1, sigma_y=2, size=5)  # type: Gaussian
EPOCH = datetime(2020, 1, 1)
camera = Camera(model=MODEL, parse_data=False, psf=PSF)


def _render_stars(camera_to_render: Camera):
    """
    Render stars in an image

    :param camera_to_render: The camera to render the stars into
    """
    star_ids = []

    camera_to_render.only_long_on()

    ii = 1

    psf = copy(PSF)
    for _, image_to_render in camera_to_render:
        start = time.time()
        # make the star_id object to get the star locations
        star_id = StarID(MODEL, catalogue=GIANTCatalogue(), max_magnitude=5.5,
                     a_priori_rotation_cat2camera=image_to_render.rotation_inertial_to_camera,
                     camera_position=image_to_render.position, camera_velocity=image_to_render.velocity)

        star_id.project_stars(epoch=image_to_render.observation_date, temperature=image_to_render.temperature)
        star_ids.append(star_id)

        drows, dcols = np.meshgrid(np.arange(-10, 10), np.arange(-10, 10), indexing='ij')

        # set the reference magnitude to be 2.5, which corresponds to a dn of 2**14
        mref = 2.5
        inten_ref = 2 ** 14

        for ind, point in enumerate(star_id.queried_catalogue_image_points.T):
            rows = np.round(drows + point[1]).astype(int)
            cols = np.round(dcols + point[0]).astype(int)

            valid_check = (rows >= 0) & (rows < MODEL.n_rows) & (cols >= 0) & (cols < MODEL.n_cols)

            if valid_check.any():
                use_rows = rows[valid_check]
                use_cols = cols[valid_check]

                # compute the intensity
                inten = 10 ** (-(star_id.queried_catalogue_star_records.iloc[ind].mag - mref) / 2.5) * inten_ref
                psf.amplitude = inten
                psf.centroid_x = point[0]
                psf.centroid_y = point[1]

                np.add.at(image_to_render, (use_rows, use_cols), psf.evaluate(use_cols, use_rows))
        print('rendered stars for image {} of {} in {:.3f} secs'.format(ii, sum(camera_to_render.image_mask),
                                                                        time.time() - start))

        ii += 1

    camera_to_render.all_on()
    return star_ids

# sun = SceneObject(Point(np.zeros(3)), position_function=_truth_sun_position,
#                   orientation_function=sun_orientation, name='Sun')

# scene = Scene(light_obj=sun)

scene = Scene()

radec_steps = 100
dec_steps = 1
right_ascention = np.linspace(-np.pi/2, np.pi/2, radec_steps)
step = 1.0
declination = np.linspace(89.0, 89.0, dec_steps)
# declination = np.arange(-90, 90, step)
camera_position = np.array([0, 0, 0])
camera_velocity = np.array([0, 0, 0])

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
color = (255, 255, 255)  # White color
thickness = 2
line_type = cv2.LINE_AA


radec_titles = []
i = 0

for ra in right_ascention:
    for dec in declination:
        # For each orientation render
        radec_titles.append(f"ra, dec = {ra:.2f}, {dec:.2f}")
        # R = SciRot.from_euler('ZYX', [-ra, dec, 0.0], degrees=False)
        # T = rotation_matrix(ra, dec)
        T = rot_mat_2(ra, dec)
        true_att = Rotation()
        true_att.matrix = T

        img = np.zeros((width, height))
        oimg = OpNavImage(img, observation_date=EPOCH, exposure=5, exposure_type='long',
                          temperature=0, rotation_inertial_to_camera=true_att, position=camera_position,
                          velocity=camera_velocity, saturation=2**16)
        camera.add_images([oimg])

star_ids_out = _render_stars(camera)

radius_scale_factor = 8/768
thick_scale_factor = 1/768
color = (255, 0, 0)
datestr = EPOCH.strftime("%Y-%m-%d_%H-%M-%S")
output_dir = os.path.join(os.getcwd(), "results", datestr)

if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=False)

for ii, img in enumerate(tqdm(camera.images)):
    star_id = star_ids_out[ii]
    out_file = os.path.join(output_dir, f"frame_{i:03d}.npy")
    frame_data = {
        "img": img,
        "queried_catalogue_image_points": star_id.queried_catalogue_image_points,
        "queried_catalogue_star_records": star_id.queried_catalogue_star_records
    }
    np.savez(out_file, **frame_data)
    _ = centroiding_pipeline(image_path, args, stel)

