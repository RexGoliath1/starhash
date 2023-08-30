import matplotlib.pyplot as plt
import json
import cv2
import os
import numpy as np

basedir = "results/20230830-080838"
image = basedir + "/images/stellarium-002.png"
img = cv2.imread(filename=image, flags=cv2.COLOR_BGR2RGB)

file = basedir + "/img_coords/1.json"
with open(file, 'r') as fp:
    jl = json.load(fp)
    positions = {k: np.array(v) for k, v in jl.items()}

# Text settings
FONT_ENABLED = False
window_name = 'Image'
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = .1
color = (255, 0, 0)
thickness = 1

for star in positions.keys():
    pos = positions[star].astype(np.int32)
    postup = (pos[0][0], pos[1][0])
    cv2.circle(img=img, center=postup, radius=3, color=(255, 0, 0), thickness=1)
    if FONT_ENABLED:
        img = cv2.putText(img, star, postup, font, fontScale, color, thickness, cv2.LINE_AA)

plt.imshow(img)
plt.title(f"Star Field with {len(positions.keys())} HIP Stars")
plt.show()
