import matplotlib.pyplot as plt
import json
import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
from glob import iglob
import re
from astropy import units as u
from astropy.coordinates import SkyCoord

DEBUG_PROP_CATALOG = True
DEBUG_STELLAR_COORDS = False
DEBUG_STELLAR_POSITIONS = False

def xyz_to_coords(E, K, width, height):
    """ Function to convert proper motion contents to image coordinates """
    coords = {}
 

    data = pm_df[["x", "y", "z"]].to_numpy()
    data = np.hstack([data, np.ones([data.shape[0], 1])])
    x_c = np.matmul(E, data.T)
    x_p = np.matmul(K, x_c)
    uv = (x_p[:2, :] / x_p[2, :])


    in_bounds = (x_p[2] > 0) & (uv[0] >= 0) & (uv[0] < width) & (uv[1] >= 0) & (uv[1] < height)
    uv_in_bounds = uv[:, in_bounds]
    object_names = cat_df.iloc[in_bounds]['HIP'].apply(lambda x: f'HIP {x}').tolist()
    coords = {name: uv_in_bounds[:,i].tolist() for i, name in enumerate(object_names)}

    return coords

# Text settings
FONT_ENABLED = True
window_name = 'Image'
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = .5
color = (255, 0, 0)
hipcolor = (0, 255, 0)
thickness = 1
pattern = r'stellarium-(\d+)\.png'
# radius_scale_factor = 3/768
radius_scale_factor = 8/768
pm_radius_scale_factor = 20/768
thick_scale_factor = 1/768

### Get datefolder for plotting
results_folder = os.path.join(os.path.dirname(__file__), 'results')
default_date = datetime.strptime('20181110-105531', "%Y%m%d-%H%M%S")
most_recent = default_date

# TODO: Replace with something more automatic
input_cat_file = f"/Users/stevengonciar/git/starhash/results/input_catalog.csv"
cat_df = pd.read_csv(input_cat_file, header=None)
cat_df.columns = ["RAJ2000", "DEJ2000", "HIP","RArad","DErad","Plx","pmRA","pmDE","Hpmag","B-V"]

pm_file = f"/Users/stevengonciar/git/starhash/results/proper_motion_data.csv"
pm_df = pd.read_csv(pm_file, header=None)
pm_df.columns = ["x", "y", "z"]

for res_folder in os.listdir(results_folder):
    try:
        date = datetime.strptime(res_folder, "%Y%m%d-%H%M%S")
    except:
        continue

    if date is None:
        continue
    elif date > most_recent:
        most_recent = date

if most_recent == default_date:
    raise Exception("No recent folders found!")

print(f"Evaluating date {most_recent}")

mr_folder = most_recent.strftime("%Y%m%d-%H%M%S")
basedir = os.path.join(results_folder, mr_folder)
image_dir = os.path.join(basedir, "images")
coords_dir = os.path.join(basedir, "img_coords")
ra_dec_dir = os.path.join(basedir, "ra_dec")
outputdir = os.path.join(basedir, "overlay_images")
os.makedirs(outputdir, exist_ok=True)

### Plot everything and output
for image in sorted(iglob(os.path.join(image_dir, "stellarium*.png"))):
    image_name = os.path.basename(image)
    output_file = os.path.join(outputdir, image_name)
    img = cv2.imread(filename=image, flags=cv2.COLOR_BGR2RGB)
    height, width, channels = img.shape

    match = re.match(pattern, image_name)

    if match:
        number = int(match.group(1))
    else:
        continue

    plt.figure(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)  # Adjust dpi as needed
    
    file = os.path.join(ra_dec_dir, f"{number - 1}_ra_dec.json")
    with open(file, 'r') as fp:
        jl = json.load(fp)
        coord = SkyCoord(ra=jl["ra"]*u.radian, dec=jl["dec"]*u.radian, frame='icrs')
        coord_hms_dms = coord.to_string('hmsdms', precision=3, pad=True, alwayssign=True)
        E = jl["E"]
        K = jl["K"]
        E = np.array(E)
        K = np.array(K)
        fov = jl["fov"]
        ang_thresh = fov / width * 3600 # Two Pixels

    # Stellarium Stars
    if DEBUG_STELLAR_COORDS:
        file = os.path.join(coords_dir, f"{number - 1}.json")
        with open(file, 'r') as fp:
            jl = json.load(fp)
            coords = {k: np.array(v) for k, v in jl.items()}

        for star in coords.keys():
            pos = coords[star].astype(np.int32)
            postup = (pos[0], pos[1])
            radius_size = int(radius_scale_factor * img.shape[0])
            thick_size = int(thick_scale_factor * img.shape[0])
            cv2.circle(img=img, center=postup, radius=radius_size, color=color, thickness=thick_size)
            if FONT_ENABLED:
                img = cv2.putText(img, star, postup, font, fontScale, color, thickness, cv2.LINE_AA)

        plt.title(f"Star Field at: {coord_hms_dms} with {len(coords.keys())} HIP Stars")

    # Debugging.. For every stellarium star, check the xyz postion from HIP
    if DEBUG_STELLAR_POSITIONS:
        file = os.path.join(coords_dir, f"positions_{number - 1}.json")
        with open(file, 'r') as fp:
            jl = json.load(fp)
            stellar_pos = {k: np.array(v) for k, v in jl.items()}

        num_stars = len(stellar_pos.keys())
        num_stars_ae = 0 # Angle Error Stars
        for row in range(0, pm_df.shape[0]):
            # TODO: Replace with star names to check
            star = f'HIP {cat_df["HIP"].loc[row]}'
            if star in sorted(stellar_pos.keys()):
                pos = stellar_pos[star]
                pos = pos / np.linalg.norm(pos)
                hpos = np.array(pm_df[["x", "y", "z"]].loc[row])
                hpos = hpos / np.linalg.norm(hpos)
                dangle = np.abs(np.arccos(np.dot(pos, hpos)) * 3600 * 180.0 / np.pi)
                if dangle > ang_thresh:
                    num_stars_ae += 1
                    print(f"{star} Deltas Angle (arcsec): {dangle}")

        print(f"Star Outage: {num_stars_ae}/{num_stars} = {100.0 * num_stars_ae/num_stars:.2f}%")

    # Propagated Catalog Stars
    if DEBUG_PROP_CATALOG:
        pm_pos = xyz_to_coords(E, K, width, height)
        for star in pm_pos.keys():
            pos = pm_pos[star]
            postup = (np.int32(pos[0]), np.int32(pos[1]))
            radius_size = int(pm_radius_scale_factor * img.shape[0])
            thick_size = int(thick_scale_factor * img.shape[0])
            cv2.circle(img=img, center=postup, radius=radius_size, color=hipcolor, thickness=thick_size)
            if FONT_ENABLED:
                img = cv2.putText(img, star, postup, font, fontScale, hipcolor, thickness, cv2.LINE_AA)
        plt.title(f"Star Field at: {coord_hms_dms} with {len(pm_pos.keys())} HIP Stars")

    plt.imshow(img)
    plt.savefig(os.path.join(outputdir, image_name), bbox_inches='tight')
    plt.close()

