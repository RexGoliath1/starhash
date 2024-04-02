import re
import os
import numpy as np
import json
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

DEFAULT_DATE = datetime.strptime('20181110-105531', "%Y%m%d-%H%M%S")

class StellarUtils():
    """ Stellarium Output Utilities for loading images / camera data """
    def __init__(self, input_path, dt = None):
        self._load_config(input_path, dt)
    def _find_latest(self, input_path, most_recent=DEFAULT_DATE):
        for res_folder in os.listdir(input_path):
            try:
                date = datetime.strptime(res_folder, "%Y%m%d-%H%M%S")
            except:
                continue

            if date is None:
                continue
            elif date > most_recent:
                most_recent = date

        if most_recent == DEFAULT_DATE:
            raise Exception("No recent folders found!")
        return most_recent.strftime("%Y%m%d-%H%M%S")

    def _load_config(self, input_path: str, dt = None, coords_pattern = "_data"):
        if dt is None:
            mr_path = self._find_latest(input_path)
        else:
            assert(type(dt) == datetime)
            mr_path = self._find_latest(input_path, dt)

        self.coords_path= os.path.join(input_path, mr_path, "data")
        self.image_path= os.path.join(input_path, mr_path, "images")
        self.coords_pattern = coords_pattern

    def get_stel_data(self, image_path):
        img_name = os.path.basename(image_path)

        match = re.search(r"\d+", img_name)
        if (match == None):
            print(f"No matching image number found in {img_name}")
            return False

        number = int(match.group())

        coord_file = os.path.join(self.coords_path, str(number) + self.coords_pattern + ".json")

        if not os.path.isfile(coord_file):
            print(f"No matching coordinate number found for {coord_file}")
            return False

        with open(coord_file, 'r') as fp:
            jl = json.load(fp)
            self.E = np.array(jl["E"])
            self.K = np.array(jl["K"])
            self.width  = jl["width"]
            self.height = jl["height"]
            self.fx = self.K[0, 0]
            self.fy = self.K[1, 1]
            self.cx = self.K[0, 2]
            self.cy = self.K[1, 2]
            self.hfov = 2 * np.arctan2(self.width / 2.0, self.fx)
            self.vfov = 2 * np.arctan2(self.height / 2.0, self.fy)
            self.dfov = np.sqrt(self.hfov**2 + self.vfov**2)
            self.hifov = self.hfov / self.width
            self.vifov = self.vfov / self.height
            self.K_inv = np.linalg.inv(self.K)
            assert (np.sum(np.linalg.inv(self.K_inv) - self.K) < 1e-12)
            self.T_cam_to_j2000 = self.E[:,:3]
            truth_stars = jl["stars"]

        num_stars = len(truth_stars.keys())
        star_names = list(truth_stars.keys())
        truth_coords = np.zeros([2, num_stars])
        truth_vectors = np.zeros([3, num_stars])
        truth_body_vectors = np.zeros([3, num_stars])

        truth_coords_list = []
        for ii in range(num_stars):
            star_name = star_names[ii]
            truth_coords[:, ii] = truth_stars[star_name]["pixel"]
            truth_coords_list.append(truth_stars[star_name]["pixel"])
            truth_vectors[:, ii] = truth_stars[star_name]["vec_inertial"]
            truth_body_vectors[:, ii] = truth_stars[star_name]["vec_camera"]

        self.truth_coords = truth_coords
        self.truth_vectors = truth_vectors
        self.truth_body_vectors = truth_body_vectors
        self.truth_coords_list = truth_coords_list
        
        return True
    
