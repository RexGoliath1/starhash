import matplotlib.pyplot as plt
import json
import cv2
import os
import numpy as np
from datetime import datetime
from glob import iglob
import re

# Text settings
FONT_ENABLED = False
window_name = 'Image'
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = .5
color = (255, 0, 0)
thickness = 1
pattern = r'stellarium-(\d+)\.png'
radius_scale_factor = 3/768
thick_scale_factor = 1/768

### Get datefolder for plotting
results_folder = os.path.join(os.path.dirname(__file__), 'results')
default_date = datetime.strptime('20181110-105531', "%Y%m%d-%H%M%S")
most_recent = default_date

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

mr_folder = most_recent.strftime("%Y%m%d-%H%M%S")
basedir = os.path.join(results_folder, mr_folder)
image_dir = os.path.join(basedir, "images")
coords_dir = os.path.join(basedir, "img_coords")
outputdir = os.path.join(basedir, "overlay_images")
os.makedirs(outputdir, exist_ok=True)

### Plot everything and output
for image in iglob(os.path.join(image_dir, "stellarium*.png")):
    image_name = os.path.basename(image)
    output_file = os.path.join(outputdir, image_name)
    img = cv2.imread(filename=image, flags=cv2.COLOR_BGR2RGB)

    match = re.match(pattern, image_name)

    if match:
        number = int(match.group(1))
    else:
        continue

    file = os.path.join(coords_dir, f"{number - 1}.json")
    with open(file, 'r') as fp:
        jl = json.load(fp)
        positions = {k: np.array(v) for k, v in jl.items()}

    for star in positions.keys():
        pos = positions[star].astype(np.int32)
        postup = (pos[0], pos[1])
        radius_size = int(radius_scale_factor * img.shape[0])
        thick_size = int(thick_scale_factor * img.shape[0])
        cv2.circle(img=img, center=postup, radius=radius_size, color=(255, 0, 0), thickness=thick_size)
        if FONT_ENABLED:
            img = cv2.putText(img, star, postup, font, fontScale, color, thickness, cv2.LINE_AA)

    #plt.figure(dpi=150)
    #plt.figure(dpi=150)
    plt.figure(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)  # Adjust dpi as needed
    plt.imshow(img)
    plt.title(f"Star Field with {len(positions.keys())} HIP Stars")
    plt.savefig(os.path.join(outputdir, image_name), bbox_inches='tight')
