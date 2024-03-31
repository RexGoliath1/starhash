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
from time import time
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
REPO_DIRECTORY = os.path.dirname(CURRENT_DIRECTORY)
IMAGE_DIRECTORY = os.path.join(REPO_DIRECTORY, "stellarscript", "results")
OUTPUT_DIRECTORY = os.path.join(CURRENT_DIRECTORY, "results")
DEFAULT_CONFIG = os.path.join(REPO_DIRECTORY, "data", "solver.yaml")
DEFAULT_CATALOG= os.path.join(REPO_DIRECTORY, "results", "scg", "output.h5")
_MAGIC_RAND = 2654435761

def modular_exponentiation(base, exponent, modulus):
    result = 1
    base %= modulus

    # Determine the number of iterations based on the number of bits in 'exponent'
    iterations = exponent.bit_length()

    for _ in range(iterations):
        if exponent & 1:
            result = (result * base) % modulus
        base = (base * base) % modulus
        exponent >>= 1

    return result

def modular_multiplication(a, b, modulus):
    result = 0
    a %= modulus

    # Determine the number of iterations based on the number of bits in 'b'
    b = int(b)  # Convert to Python int
    iterations = b.bit_length()

    for _ in range(iterations):
        if b & 1:  # If the least significant bit of 'b' is 1
            result = (result + a) % modulus
        a = (2 * a) % modulus
        b >>= 1  # Right shift 'b' to divide by 2

    return result

def _key_to_index(key, bin_factor, max_index):
    """Get hash index for a given key."""
    index_sum = 0
    for i, val in enumerate(key):
        # Using custom modular_exponentiation function
        term = val * modular_exponentiation(bin_factor, i, max_index) % max_index
        index_sum = (index_sum + term) % max_index

    # Use modular_multiplication for the final multiplication by the magic number
    final_index = modular_multiplication(index_sum, _MAGIC_RAND, max_index)

    return final_index

class StarSolver():
    def __init__(self, args):
        self.args = self.process_params(args)
        self.args = args # TODO: Incredibly lazy, redo
        self.stel = StellarUtils(args.input_path)
        self._load_catalog(args.catalog_file)
        self.t1 = time()

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

    def pix_to_vec(self, centroids, K_inv):
        """ pixel to vector in camera frame """
        uv = np.hstack((centroids, np.ones((centroids.shape[0], 1))))
        # vec = uv @ K_inv.T
        vec = (K_inv @ uv.T).T

        # This technically doesn't happen if K is being optimized
        # assert(all(np.abs(vec[:, 0]) <= 1.0))
        # assert(all(np.abs(vec[:, 1]) <= 1.0))

        vec[:,2] = np.sqrt(1- vec[:,0]**2 - vec[:,1]**2)
        norm = np.sqrt(np.einsum('ij,ij->i', vec, vec))
        norm = norm[:, np.newaxis]
        # assert(all(norm > 0)) # Hard to assert with leastsq
        vec = vec * (1 / norm)
        return vec

    def vec_to_pix(self, vectors, K):
        """ Unit test to check math: Convert camera vector to pixel """
        uv = (K @ vectors.T).T

        if any(uv[:, -1] <= 0):
            raise Exception("Something wrong. Camera space has negative z")

        xy = uv[:, :-1] / uv[:, -1, np.newaxis]

        # For debugging, print out stars within frame
        if any(0 > xy[:, 0]) or any(xy[:, 0] >= self.stel.width):
            raise Exception("Something wrong. Camera x pixel out of frame")
        if any(0 > xy[:, 1]) or any(xy[:, 1] >= self.stel.height):
            raise Exception("Something wrong. Camera y pixel out of frame")

        return xy

    def _sort_vectors(self, vectors):
        """ Sort vectors wrt radius from vector centroid (TODO: Is this good enough, may require multiples) """
        centroid = np.mean(vectors, axis=0)
        radii = [np.linalg.norm(vector - centroid) for vector in vectors]
        radii_idx = np.argsort(radii, kind="mergesort")
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

    # def _key_to_index(self, key, bin_factor, max_index):
    #     """Get hash index for a given key."""
    #     # Get key as a single integer
    #     # index = sum(int(val) * int(bin_factor)**i for (i, val) in enumerate(key))
    #     index = list(int(val) * int(bin_factor)**i for (i, val) in enumerate(key))
    #     index_sum = sum(index)
    #     # Randomise by magic constant and modulo to maximum index
    #     # print(f"index = {index}")
    #     return (index_sum * _MAGIC_RAND) % max_index
    
    def _get_edges(self, vectors):
        """ Return edge ratios smallest to largest """
        vec_pairs = itertools.combinations(vectors, 2)
        edges = [np.linalg.norm(np.subtract(*star_pair)) for star_pair in vec_pairs]
        return np.sort(edges, kind="mergesort")

    def _get_angles(self, vectors):
        """ Return edge ratios smallest to largest """
        vec_pairs = itertools.combinations(vectors, 2)
        angles = [np.arccos(np.dot(*star_pair)) for star_pair in vec_pairs]
        return np.sort(angles, kind="mergesort")

    def get_closest_points(self, truth, measured, angle_threshold_sf=3.0):
        """ Use KDTree/Hungarian to map centroids to truth """
        angle_threshold = angle_threshold_sf * max(self.stel.hifov, self.stel.vifov)
        max_cost = angle_threshold * 1000
        assert(angle_threshold < max_cost)
        tree = cKDTree(truth)
        idx = tree.query_ball_point(x=measured, r=angle_threshold, p=2)
        cost_matrix = np.full((len(measured), len(truth)), max_cost)

        if idx is None:
            raise Exception("Shouldn't happen unless centroiding went bad")

        for i, indices in enumerate(idx):
            for truth_index in indices:
                distance = np.linalg.norm(np.array(truth[truth_index]) - np.array(measured[i]))
                cost_matrix[i, truth_index] = distance

        measured_indicies, truth_indices = linear_sum_assignment(cost_matrix)

        valid_assignments = cost_matrix[measured_indicies, truth_indices] < pix_thresh
        measured_indicies = measured_indicies[valid_assignments]
        truth_indices = truth_indices[valid_assignments]
        return measured_indicies, truth_indices

    def fov_error(self, params, image_centroids, cat_edges):
        """ Optimization func for determining FOV """
        fx, fy = params
        cx = self.stel.cx
        cy = self.stel.cy

        # TODO: Tweak leastsq to prevent this
        if (np.any([np.isnan(param) for param in params])):
            return np.inf * cat_edges

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        K_inv = np.linalg.inv(K)

        if (np.any([np.isnan(coeff) for coeff in K_inv])):
            return np.inf * cat_edges

        star_vectors = self.pix_to_vec(image_centroids, K_inv)
        star_edges = self._get_edges(star_vectors)
        # star_angles = self._get_angles(star_vectors)
        return star_edges - cat_edges

    def solve_from_centroids(self, centroids):
        """ Solve lost in space from centroid data """
        pattern_stars = args.pattern_stars
        gen_count = 0
        total_check_count = 0
        # for pattern_centroids in self._pattern_generator(centroids, self.args.pattern_size):
        for pattern_centroids in self._pattern_generator(centroids[:pattern_stars], self.args.pattern_size):
            gen_count += 1
            self.pattern_centroids = pattern_centroids

            # pattern_centroids = np.array([self.stel.width/2 -20, self.stel.height/2 - 20])
            # pattern_centroids = np.array([100, self.stel.height/2 - 20])
            # pattern_centroids = pattern_centroids[np.newaxis, :]

            vectors = self.pix_to_vec(pattern_centroids, self.stel.K_inv)
            rev_centroids = self.vec_to_pix(vectors, self.stel.K)
            # diff_centroids = pattern_centroids - rev_centroids
            # assert(all(diff_centroids < 1e-12))
            edges = self._get_edges(vectors)
            eratios = edges[:-1] / edges[-1]

            angles = self._get_angles(vectors)
            aratios = angles[:-1] / angles[-1]
            ratios = np.concatenate([eratios, aratios])

            # TODO: Implement all of the crazy hash processing.
            #hash_code_space = [range(max(int(low), 0), min(int(high) + 1, args.pattern_bins)) for (low, high) 
            #    in zip((ratios - args.pattern_max_error) * args.pattern_bins, (ratios + args.pattern_max_error) * args.pattern_bins)]

            hash_code_space = [range(max(low, 0), min(high+1, args.pattern_bins)) for (low, high)
                in zip(((ratios - args.pattern_max_error) * args.pattern_bins).astype(int),
                       ((ratios + args.pattern_max_error) * args.pattern_bins).astype(int))]

            # hash_code_space = [range(max(low, 0), min(high+1, args.pattern_bins)) for (low, high)
            #                    in zip(((ratios - args.pattern_max_error) * self.args.pattern_bins).astype(np.int64),
            #                           ((ratios + args.pattern_max_error) * self.args.pattern_bins).astype(np.int64))]

            all_codes = itertools.product(*hash_code_space)
            for code in set(tuple(all_codes)):
                code = tuple(code)
                index = _key_to_index(code, self.args.pattern_bins, self.pattern_catalog.shape[0])
                matches = self._get_at_index(index, self.pattern_catalog)
                if matches is None or len(matches) == 0:
                    #print(f"No matches found for {code}")
                    continue

                for match_row in matches:
                    total_check_count += 1
                    cat_vectors = self.star_table[match_row]
                    cat_edges = self._get_edges(cat_vectors)
                    cat_angles = self._get_angles(cat_vectors)
                    cat_eratios = cat_edges[:-1] / cat_edges[-1]
                    cat_aratios = cat_angles[:-1] / cat_angles[-1]
                    cat_ratios = np.concatenate([cat_eratios, cat_aratios])

                    # if np.sum(cat_edges - edges) > self.args.max_edge_diff:
                    #     print("Skipping for now...")

                    if (np.any(np.abs(cat_eratios - eratios) > args.pattern_max_error)):
                        continue

                    # Projection optimization that minimizes edge errors
                    # estimated_params = self.stel.cx, self.stel.cy, self.stel.fx, self.stel.fy
                    estimated_params = self.stel.fx, self.stel.fy
                    assert(not np.any([np.isnan(param) for param in estimated_params]))
                    params = scipy.optimize.leastsq(self.fov_error, estimated_params, args=(pattern_centroids, cat_edges), full_output=True)

                    # # TODO: Tweak leastsq to prevent this
                    if (np.any([np.isnan(param) for param in params[0]])):
                        continue
                    fx, fy = params[0]
                    K_opt = np.array([[fx, 0, self.stel.cx], [0, fy, self.stel.cy], [0, 0, 1]])

                    # Debug
                    K_opt = self.stel.K

                    fx = K_opt[0, 0]
                    fy = K_opt[1, 1]
                    hfov = np.arctan2(self.stel.width / 2.0, fx)
                    vfov = np.arctan2(self.stel.height/ 2.0, fy)
                    hfov_error = np.rad2deg(np.abs(hfov - self.stel.hfov))
                    vfov_error = np.rad2deg(np.abs(vfov - self.stel.vfov))

                    # if self.args.fov_max_error is not None and (hfov_error > self.args.fov_max_error or vfov_error > self.args.fov_max_error):
                    #     # print(f"Estimated HFOV / VFOV Error beyond max acceptable: {hfov_error:.2f}, {vfov_error:.2f}")
                    #     continue

                    # Set up new intrinsics
                    self.K = K_opt
                    K_inv = np.linalg.inv(self.K)
                    pattern_vectors = self.pix_to_vec(pattern_centroids, K_inv)
                    pattern_vectors = self._sort_vectors(pattern_vectors)
                    cat_vectors = self._sort_vectors(cat_vectors)
                    C_opt, q_opt = quest(pattern_vectors.T, cat_vectors.T)
                    rot_vec_error = rotation_vector_from_matrices(self.stel.T_cam_to_j2000, C_opt)
                    # centroid_vectors = self.pix_to_vec(centroids, K_inv)
                    # centroid_inertial_vectors = C_opt @ centroid_vectors
                    # boresight_vector = C_opt[0, :]
                    # self._get_nearby_stars(boresight_vector, )

                    rv_deg = np.degrees(rot_vec_error)
                    rv_as = rv_deg * 3600
                    rv_mas = 1000 * rv_as

                    if (np.linalg.norm(rv_deg) < 0.1):
                        print(f"Took {time() - self.t1} seconds")
                        print(f"rv_mag_mas: {np.linalg.norm(rv_mas):.2f} rot_vec_error: {rv_mas} (Milli Arc-Seconds)")
                        print(f"rv_mag_as: {np.linalg.norm(rv_as):.2f} rot_vec_error: {rv_as} (Arc-Seconds)")
                        print(f"rv_mag_deg: {np.linalg.norm(rv_deg):.2f} rot_vec_error: {rv_deg} (Degrees)")
                        print(f"TODO: Star Error Statistics")
                        return



    def solver_pipeline(self):
        for image_path in iglob(os.path.join(self.stel.image_path, self.args.image_pattern)):
            print(f"Solving for {image_path}")
            self.t1 = time()
            if not self.stel.get_stel_data(image_path):
                continue

            centroid_params = centroiding_pipeline(image_path, args, self.stel)
            star_centroids = centroid_params["star_centroids"]

            self.solve_from_centroids(star_centroids)

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
