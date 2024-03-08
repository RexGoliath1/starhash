from centroiding import centroiding_pipeline
import argparse
import os
from stellar_utils import StellarUtils
from glob import iglob
import numpy as np
import itertools
import h5py
import scipy
from attitude_determination import quest
from transform_utils import rotation_vector_from_matrices
import sys
import yaml
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
REPO_DIRECTORY = os.path.dirname(CURRENT_DIRECTORY)
IMAGE_DIRECTORY = os.path.join(REPO_DIRECTORY, "stellarscript", "results")
OUTPUT_DIRECTORY = os.path.join(CURRENT_DIRECTORY, "results")
DEFAULT_CONFIG = os.path.join(REPO_DIRECTORY, "data", "solver.yaml")
DEFAULT_CATALOG= os.path.join(REPO_DIRECTORY, "results", "scg", "output.h5")
_MAGIC_RAND = 2654435761

class StarSolver():
    def __init__(self, args):
        self.args = self.process_params(args)
        self.args = args # TODO: Incredibly lazy, redo
        self.stel = StellarUtils(args.input_path)
        self._load_catalog(args.catalog_file)

    def process_params(self, args):
        # Override parameters provided in the config
        if args.config_path is not None and os.path.exists(args.config_path):
            with open(args.config_path, 'r') as stream:
                config = yaml.safe_load(stream)
                for param in config.keys():
                    setattr(args, param, config[param])

        os.makedirs(args.output_path, exist_ok=True)
        assert (os.path.exists(args.input_path))
        assert (os.path.exists(args.output_path))

    def _load_catalog(self, catalog_file):
        assert(os.path.isfile(catalog_file))
        with h5py.File(catalog_file, 'r') as hf:
            self.star_table = np.array(hf.get('star_table'))
            self.pattern_catalog = np.array(hf.get('pattern_catalog'))

    def _pattern_generator(self, centroids, pattern_size):
        """Iterate over centroids in order of brightness."""
        # break if there aren't enough centroids to make even one pattern
        if len(centroids) < pattern_size:
            return

        star_centroids = np.array(centroids)
        pattern_indices = [-1] + list(range(pattern_size)) + [len(star_centroids)]

        yield star_centroids[pattern_indices[1:-1]]
        while pattern_indices[1] < len(star_centroids) - pattern_size:
            for index_to_change in range(1, pattern_size + 1):
                pattern_indices[index_to_change] += 1
                if pattern_indices[index_to_change] < pattern_indices[index_to_change + 1]:
                    break
                else:
                    pattern_indices[index_to_change] = pattern_indices[index_to_change - 1] + 1
            yield star_centroids[pattern_indices[1:-1]]

    def pix_to_vec(self, centroid, K_inv):
        """ pixel to vector in camera frame """
        uv = np.hstack((centroid, np.ones((centroid.shape[0], 1))))
        vec = uv @ K_inv
        vec[:,2] = 1- vec[:,0]**2 - vec[:,1]**2
        norm = np.sqrt(np.einsum('ij,ij->i', vec, vec))
        norm = norm[:, np.newaxis]
        vec = vec * (1 / norm)
        return vec

    def _sort_vectors(self, vectors):
        """ Sort vectors wrt radius from vector centroid (TODO: Is this good enough, may require multiples) """
        centroid = np.mean(vectors, axis=0)
        radii = [np.linalg.norm(vector - centroid) for vector in vectors]
        radii_idx = np.argsort(radii)
        return vectors[radii_idx]

    def _get_at_index(self, index, table):
        """Gets from table with quadratic probing, returns list of all matches."""
        max_ind = table.shape[0]
        found = []
        for c in itertools.count():
            i = (index + c**2) % max_ind
            if all(table[i, :] == 0):
                return found
            else:
                found.append(table[i, :].squeeze())

    def _key_to_index(self, key, bin_factor, max_index):
        """Get hash index for a given key."""
        # Get key as a single integer
        # index = sum(int(val) * int(bin_factor)**i for (i, val) in enumerate(key))
        index = list(int(val) * int(bin_factor)**i for (i, val) in enumerate(key))
        index_sum = sum(index)
        # Randomise by magic constant and modulo to maximum index
        # print(f"index = {index}")
        return (index_sum * _MAGIC_RAND) % max_index
    
    def _get_edges(self, vectors):
        """ Return edge ratios smallest to largest """
        vec_pairs = itertools.combinations(vectors, 2)
        edges = [np.linalg.norm(np.subtract(*star_pair)) for star_pair in vec_pairs]
        return np.sort(edges)

    def _get_max_edge(self, vectors):
        vec_pairs = itertools.combinations(vectors, 2)
        edges = [np.linalg.norm(np.subtract(*star_pair)) for star_pair in vec_pairs]
        edges = np.sort(edges)
        return edges[-1]

    def fov_error(self, params, image_centroids, cat_edges):
        K = params
        # cx, cy, fx, fy = params
        # K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        K_inv = np.linalg.inv(K)
        star_vectors = self.pix_to_vec(image_centroids, K_inv)
        star_edges = self._get_edges(star_vectors)
        return star_edges - cat_edges

    def solve_from_centroids(self, centroids):
        """ Solve lost in space from centroid data """
        pattern_stars = args.pattern_stars
        gen_count = 0
        for pattern_centroids in self._pattern_generator(centroids[:pattern_stars], self.args.pattern_size):
            gen_count += 1
            self.pattern_centroids = pattern_centroids
            vectors = self.pix_to_vec(pattern_centroids, self.stel.K_inv)
            edges = self._get_edges(vectors)
            ratios = edges[:-1] / edges[-1]
            hash_code_space = [range(max(low, 0), min(high+1, args.pattern_bins)) for (low, high)
                               in zip(((ratios - args.pattern_max_error) * self.args.pattern_bins).astype(int),
                                      ((ratios + args.pattern_max_error) * self.args.pattern_bins).astype(int))]

            all_codes = itertools.product(*hash_code_space)
            for code in set(tuple(all_codes)):
                code = tuple(code)
                index = self._key_to_index(code, self.args.pattern_bins, self.pattern_catalog.shape[0])
                matches = self._get_at_index(index, self.pattern_catalog)
                if matches is None or len(matches) == 0:
                    #print(f"No matches found for {code}")
                    continue

                for match_row in matches:
                    cat_vectors = self.star_table[match_row]
                    cat_edges = self._get_edges(cat_vectors)
                    cat_ratios = cat_edges[:-1] / cat_edges[-1]
                    if (np.any(np.abs(cat_ratios - ratios) > args.pattern_max_error)):
                        continue

                    # Projection optimization that minimizes edge errors
                    K = scipy.optimize.minimize(self.fov_error, self.stel.K, args=(pattern_centroids, cat_edges))

                    K_opt = K[0][0]
                    fx = K_opt[0, 0]
                    fy = K_opt[1, 1]
                    hfov = np.arctan2(self.stel.width / 2.0, fx)
                    vfov = np.arctan2(self.stel.height/ 2.0, fy)
                    hfov_error = np.abs(hfov - self.stel.hfov)
                    vfov_error = np.abs(vfov - self.stel.vfov)

                    if self.args.fov_max_error is not None and hfov_error > self.args.fov_max_error and vfov_error > self.args.fov_max_error:
                        print(f"Estimated HFOV / VFOV Error beyond max acceptable: {hfov_error:.2f}, {vfov_error:.2f}")

                    # Set up new intrinsics
                    self.K = K_opt
                    pattern_vectors = self.pix_to_vec(pattern_centroids, self.K)
                    pattern_vectors = self._sort_vectors(pattern_vectors)
                    cat_vectors = self._sort_vectors(cat_vectors)
                    C_opt, q_opt = quest(pattern_vectors, cat_vectors)
                    rot_vec_error = rotation_vector_from_matrices(self.stel.T_cam_to_j2000, C_opt)
                    rv_deg = np.degrees(rot_vec_error)
                    rv_as = rv_deg * 3600
                    rv_mas = 1000 * rv_as

                    print(f"rv_mag_mas: {np.linalg.norm(rv_mas):.2f} rot_vec_error: {rv_mas} (Milli Arc-Seconds)")
                    print(f"rv_mag_as: {np.linalg.norm(rv_as):.2f} rot_vec_error: {rv_as} (Arc-Seconds)")
                    print(f"rv_mag_deg: {np.linalg.norm(rv_deg):.2f} rot_vec_error: {rv_deg} (Degrees)")
                    print(f"TODO: Star Error Statistics")



    def solver_pipeline(self):
        for image_path in iglob(os.path.join(self.stel.image_path, self.args.image_pattern)):
            if not self.stel.get_stel_data(image_path):
                continue

            all_centroids = centroiding_pipeline(image_path, args)
            self.solve_from_centroids(all_centroids)

if __name__ == "__main__":
    print("Starting....")
    parser = argparse.ArgumentParser(description="Overlay OpenCV blob detection on an image")
    parser.add_argument("--input_path",     default=IMAGE_DIRECTORY,    type=str, help="")
    parser.add_argument("--output_path",    default=OUTPUT_DIRECTORY,   type=str, help="")
    parser.add_argument("--catalog_file",   default=DEFAULT_CATALOG,    type=str, help="")
    parser.add_argument("--config_path",    default=DEFAULT_CONFIG,     type=str, help="")
    args = parser.parse_args()

    s = StarSolver(args)
    s.solver_pipeline()
