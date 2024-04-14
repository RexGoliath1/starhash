from centroiding import centroiding_pipeline
import argparse
import os
from stellar_utils import StellarUtils
from glob import glob
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
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
REPO_DIRECTORY = os.path.dirname(CURRENT_DIRECTORY)
IMAGE_DIRECTORY = os.path.join(REPO_DIRECTORY, "stellarscript", "results")
OUTPUT_DIRECTORY = os.path.join(CURRENT_DIRECTORY, "results")
DEFAULT_CONFIG = os.path.join(REPO_DIRECTORY, "data", "solver.yaml")
DEFAULT_CATALOG = os.path.join(REPO_DIRECTORY, "results", "scg", "output.h5")
DEFAULT_COARSE = os.path.join(REPO_DIRECTORY, "results", "scg", "coarse_sky_map.yaml")
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

def _key_to_index_edges(key, bin_factor, max_index):
    """Get hash index for a given key."""
    # Get key as a single integer
    # index = sum(int(val) * int(bin_factor)**i for (i, val) in enumerate(key))
    index = list(int(val) * int(bin_factor)**i for (i, val) in enumerate(key))
    index_sum = sum(index)
    # Randomise by magic constant and modulo to maximum index
    # print(f"index = {index}")
    return (index_sum * _MAGIC_RAND) % max_index

class StarSolver():
    def __init__(self, args):
        self.args = self.process_params(args)
        self.args = args # TODO: Incredibly lazy, redo
        self.stel = StellarUtils(args.input_path)
        self._load_catalog(args.catalog_file)
        self._load_coarse(args.coarse_file)
        self._init_tree()
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

    def _load_coarse(self, coarse_file):
        assert(os.path.isfile(coarse_file))
        with open(coarse_file, 'r') as stream:
            self.coarse_map = yaml.safe_load(stream)
            print("Checking..")

    def _init_tree(self):
        self.catalog_vector_tree = cKDTree(self.star_table)
        self.cost_matrix = np.full((self.args.max_tree_measured_length, self.args.max_tree_catalog_length), np.inf)

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

    
    def _get_edges(self, vectors):
        """ Return edge ratios smallest to largest """
        vec_pairs = itertools.combinations(vectors, 2)
        edges = [np.linalg.norm(np.subtract(*star_pair)) for star_pair in vec_pairs]
        return np.sort(edges, kind="mergesort")


    def _get_edges_centroid(self, vectors):
        centroid = np.mean(vectors, axis=0)
        edges = np.linalg.norm(vectors- centroid, axis=1)
        # return np.sort(edges, kind="mergesort")
        return np.sort(edges)


    def _get_angles_centroid(self, vectors):
        centroid = np.mean(vectors, axis=0)
        edges = np.linalg.norm(vectors- centroid, axis=1)
        centroid_vectors = (vectors - centroid) / edges[:, np.newaxis]
        # edgeidx = np.argsort(edges, kind="mergesort")
        edgeidx = np.argsort(edges)
        centroid_vectors = centroid_vectors[edgeidx]

        vectors_next = np.roll(centroid_vectors, -1, axis=0)
        angles = np.einsum('ij,ij->i', centroid_vectors, vectors_next)
        directions = np.linalg.norm(np.cross(centroid_vectors, vectors_next), axis=1)

        def quadrant_calc(angle, direction):
            if angle >= 0 and direction >= 0:
                return 0.25 * (1 - angle)
            elif angle <= 0 and direction >= 0:
                return 0.25 - 0.25 * angle
            elif angle <= 0 and direction <= 0:
                return 0.5 + 0.25 * (1 + angle)
            elif angle >= 0 and direction <= 0:
                return 0.75 + 0.25 * angle

        # Vectorize the function
        vectorized_quadrant_calc = np.vectorize(quadrant_calc)

        # Apply the vectorized function to angles and directions
        return vectorized_quadrant_calc(angles, directions)

        # for ii in range(0, edges.shape[0]):
        #     next_index = (ii + 1) % edges.shape[0]
        #     v1 = centroid_vectors[ii]
        #     v2 = centroid_vectors[next_index]
        #     angle = np.dot(v1, v2)
        #     direction = np.linalg.norm(np.cross(v1, v2))
        #     # assert(np.abs(angle) <= 1)
        #     # assert(np.abs(direction) <= 1)
        #     
        #     if (angle >= 0 and direction >= 0):
        #       self.pat_angles[ii] = 0.25 * (1 - angle)
        #     elif (angle <= 0 and direction >= 0):
        #       self.pat_angles[ii] = 0.25 - 0.25 * angle
        #     elif (angle <= 0 and direction <= 0):
        #       self.pat_angles[ii] = 0.5 + 0.25 * (1 + angle)
        #     elif (angle >= 0 and direction <= 0):
        #       self.pat_angles[ii] = 0.75 + 0.25 * angle
        #     # else:
        #     #   raise Exception("Unknown centroid angle quadrant");
        # # assert(np.all(np.abs(pat_angles) <= 1))
        # assert(test == self.pat_angles)
        # return self.pat_angles


    def _get_angles(self, vectors):
        """ Return edge ratios smallest to largest """
        vec_pairs = itertools.combinations(vectors, 2)
        angles = [np.arccos(np.dot(*star_pair)) for star_pair in vec_pairs]
        return np.sort(angles, kind="mergesort")

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
        if self.args.use_star_centroid:
            star_edges = self._get_edges_centroid(star_vectors)
        else:
            star_edges = self._get_edges(star_vectors)
        # star_angles = self._get_angles(star_vectors)
        return star_edges - cat_edges

    def _get_nearby_stars(self, vector, radius):
        """Get nearby stars from coarse hash table."""
        # create list of nearby stars
        nearby_star_ids = []
        # given error of at most radius in each dimension, compute the space of hash codes
        hash_code_space = [range(max(low, 0), min(high + 1, 2 * self.args.coarse_bins)) for (low, high)
                           in zip(((vector + 1 - radius) * self.args.coarse_bins).astype(np.int32),
                                  ((vector + 1 + radius) * self.args.coarse_bins).astype(np.int32))]

        # iterate over hash code space
        for hash_code in itertools.product(*hash_code_space):
            # iterate over the stars in the given partition, adding them to
            # the nearby stars list if they're within range of the vector
            # TODO: Replace in Generator with actual numbers
            str_hash_code = str(list(hash_code))[1:-1]
            for star_ids in self.coarse_map.get(str_hash_code, []):
                star_ids = list(self.coarse_map[str_hash_code])
                for star_id in star_ids:
                    if np.dot(vector, self.star_table[star_id]) > np.cos(radius):
                        nearby_star_ids.append(star_id)

        # Shouldn't be needed
        nearby_star_ids = list(set(nearby_star_ids))
        return self.star_table[nearby_star_ids]

    def get_closest_points(self, measured, angle_thresh = None):
        """ Use KDTree/Hungarian to map centroids to truth """

        if angle_thresh is None:
            # TODO: Make better when validated outside stellarium
            angle_thresh= 10 * self.stel.difov

        euclidean_thresh = 2 * np.sin(angle_thresh / 2)
        max_cost = 2 * np.sin(self.stel.dfov / 2)
        # assert(angle_thresh < max_cost)
        measured_count = min(measured.shape[0], self.args.max_tree_measured_length)
        sub_measured = measured[:measured_count, :]
        original_meas_indices = np.arange(measured.shape[0])
        sub_cost_matrix = self.cost_matrix[:measured_count, :]

        # TODO: Make this resilient to double matches (hungarian needs constraint on already matched vectors)
        idx = self.catalog_vector_tree.query_ball_point(sub_measured, r=euclidean_thresh, p=2)

        if all([len(ii)==0 for ii in idx]):
            print("No matches for kdtree mapping. Centroiding went bad")
            return np.nan, np.nan

        for i, indices in enumerate(idx):
            for truth_index in indices:
                distance = np.arccos(np.clip(np.dot(self.star_table[truth_index], sub_measured[i]), -1.0, 1.0))
                sub_cost_matrix[i, truth_index] = distance

        measured_indicies, truth_indices = linear_sum_assignment(sub_cost_matrix)
        valid_assignments = sub_cost_matrix[measured_indicies, truth_indices] < max_cost
        measured_indicies = original_meas_indices[measured_indicies[valid_assignments]]
        truth_indices = truth_indices[valid_assignments]
        return measured_indicies, truth_indices

    def solve_from_centroids(self, centroids):
        """ Solve lost in space from centroid data """
        pattern_stars = args.pattern_stars
        gen_count = 0
        total_check_count = 0
        pattern_check_count = 0
        over_total_count = False

        for pattern_centroids in self._pattern_generator(centroids[:pattern_stars], self.args.pattern_size):
            gen_count += 1
            self.pattern_centroids = pattern_centroids
            vectors = self.pix_to_vec(pattern_centroids, self.stel.K_inv)

            if self.args.use_star_centroid:
                edges = self._get_edges_centroid(vectors)
                angles = self._get_angles_centroid(vectors)
                # All instances of vfov are hacks for now. Stellarium is only controllable through fov = vertical for now.. Change to catalog parameter
                edge_norm = np.sin(self.stel.vfov / 2)
                angle_norm = 1
            else:
                edges = self._get_edges(vectors)
                angles = self._get_angles(vectors)
                edge_norm = 2 * np.sin(self.stel.vfov / 2)
                angle_norm = self.stel.vfov

            if self.args.use_max_norm:
                aratios = angles[:-1] / angles[-1]
                eratios = edges[:-1] / edges[-1]
            else:
                eratios = edges / edge_norm
                aratios = angles / angle_norm


            if self.args.use_angles:
                ratios = np.concatenate([eratios, aratios])
            else:
                ratios = eratios

            hash_code_space = [range(max(low, 0), min(high+1, args.pattern_bins)) for (low, high)
                in zip(((ratios - args.pattern_max_error) * args.pattern_bins).astype(int),
                       ((ratios + args.pattern_max_error) * args.pattern_bins).astype(int))]

            all_codes = itertools.product(*hash_code_space)
            pattern_check_count = 0
            for code in set(tuple(all_codes)):
                code = tuple(code)

                if self.args.use_angles:
                    index = _key_to_index(code, self.args.pattern_bins, self.pattern_catalog.shape[0])
                else:
                    index = _key_to_index_edges(code, self.args.pattern_bins, self.pattern_catalog.shape[0])

                matches = self._get_at_index(index, self.pattern_catalog)
                if matches is None or len(matches) == 0:
                    #print(f"No matches found for {code}")
                    continue

                for match_row in matches:
                    pattern_check_count += 1 
                    total_check_count += 1
                    # if not over_total_count and total_check_count > 100000:
                    #     over_total_count = True
                    #     print("Oh no! Over total count and rising...")
                    #     return

                    # if pattern_check_count > 100:
                    #     print("Oh no! Over pattern count and rising...")
                    #     break

                    cat_vectors = self.star_table[match_row]

                    if self.args.use_star_centroid:
                        cat_edges = self._get_edges_centroid(cat_vectors)
                        cat_angles = self._get_angles_centroid(cat_vectors)
                        edge_norm = np.sin(self.stel.vfov / 2)
                        angle_norm = 1
                    else:
                        cat_edges = self._get_edges(cat_vectors)
                        cat_angles = self._get_angles(cat_vectors)
                        edge_norm = 2 * np.sin(self.stel.vfov / 2)
                        angle_norm = self.stel.vfov

                    if self.args.use_max_norm:
                        cat_eratios = cat_edges[:-1] / cat_edges[-1]
                        cat_aratios = cat_angles[:-1] / cat_angles[-1]
                    else:
                        cat_eratios = cat_edges / edge_norm
                        #cat_aratios = cat_angles / self.stel.vfov
                        cat_aratios = cat_angles / angle_norm

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

                    # TODO: Debug and fix least squares
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

                    rv_deg = np.degrees(rot_vec_error)
                    if (np.linalg.norm(rv_deg) > 0.1):
                        continue

                    T_j2000_to_bore = C_opt
                    T_bore_to_j2000 = np.linalg.pinv(T_j2000_to_bore)
                    # TODO: Don't understand why boresight vector is z aris of bore to j2000. Could be because camera axes flipped
                    boresight_vector = T_bore_to_j2000[:, 2]
                    if (np.dot(boresight_vector, self.stel.vec_j2000) < np.cos(np.radians(0.1))):
                        continue

                    # Temp
                    # centroid_vectors = pattern_vectors
                    # catalog_inertial_vectors = cat_vectors

                    centroid_vectors = self.pix_to_vec(centroids, K_inv)
                    centroid_inertial_vectors = (T_bore_to_j2000 @ centroid_vectors.T).T
                    # TODO: Might need to be a little higher than diagonal FOV depending on distortion
                    meas_idx, cat_idx= self.get_closest_points(centroid_inertial_vectors)

                    if any(np.isnan(cat_idx)) or any(np.isnan(meas_idx)):
                        continue
                    # Now rerun quest with larger number of matching vectors
                    C_opt, q_opt = quest(centroid_vectors[meas_idx].T, catalog_inertial_vectors[cat_idx].T)
                    rot_vec_error_refined = rotation_vector_from_matrices(self.stel.T_cam_to_j2000, C_opt)

                    rv_deg_refined = np.degrees(rot_vec_error_refined)
                    rv_as_refined = rv_deg_refined * 3600
                    rv_mas_refined = 1000 * rv_as_refined

                    if (np.linalg.norm(rv_deg) < 0.1):
                        dt = time() - self.t1
                        self.times[self.image_number] = dt
                        self.hash_checks[self.image_number] = total_check_count
                        self.star_checks[self.image_number] = gen_count
                        self.pattern_checks[self.image_number] = pattern_check_count
                        self.accuracies[self.image_number] = np.linalg.norm(rv_as)
                        print(f"Took {dt} seconds")
                        print(f"rv_mag_mas: {np.linalg.norm(rv_mas):.2f} rot_vec_error: {rv_mas} (Milli Arc-Seconds)")
                        print(f"rv_mag_as: {np.linalg.norm(rv_as):.2f} rot_vec_error: {rv_as} (Arc-Seconds)")
                        print(f"rv_mag_deg: {np.linalg.norm(rv_deg):.2f} rot_vec_error: {rv_deg} (Degrees)")
                        print(f"TODO: Star Error Statistics")
                        return

    def output_plots(self):
        """ Various Stats Plots of Solver """
        _, axs = plt.subplots(7, 1, figsize=(15, 15))
        axs[0].set_title(f"Solved for {self.num_images - np.sum(np.isnan(self.times))} / {self.num_images} images. Times: Median = {np.nanmedian(self.times):.2f}, Mean = {np.nanmean(self.times):.2f}, Max = {np.nanmax(self.times):.2f}. Min Time = {np.nanmin(self.times):.2f}, Std Dev = {np.nanstd(self.times):.2f}")
        axs[0].hist(self.times, edgecolor='black', linewidth=1.2, bins=50)
        axs[0].set_xlabel("Time (Image to Quat)")

        axs[1].set_title(f"Probe Checks: Median = {np.nanmedian(self.hash_checks)}, Mean = {np.nanmean(self.hash_checks):.2f}, Max = {np.nanmax(self.hash_checks)}. Min = {np.nanmin(self.hash_checks)}, Std Dev = {np.nanstd(self.hash_checks):.2f}")
        axs[1].hist(self.hash_checks, edgecolor='black', linewidth=1.2, bins=50)
        axs[1].set_xlabel("Hash Checks")

        axs[2].set_title(f"Star Checks: Median = {np.nanmedian(self.star_checks)}, Mean = {np.nanmean(self.star_checks):.2f}, Max = {np.nanmax(self.star_checks)}. Min = {np.nanmin(self.star_checks)}, Std Dev = {np.nanstd(self.star_checks):.2f}")
        axs[2].hist(self.star_checks, edgecolor='black', linewidth=1.2, bins=50)
        axs[2].set_xlabel("Star Checks")

        axs[3].set_title(f"Pattern Checks: Median = {np.nanmedian(self.pattern_checks)}, Mean = {np.nanmean(self.pattern_checks):.2f}, Max = {np.nanmax(self.pattern_checks)}. Min = {np.nanmin(self.pattern_checks)}, Std Dev = {np.nanstd(self.pattern_checks):.2f}")
        axs[3].hist(self.pattern_checks, edgecolor='black', linewidth=1.2, bins=50)
        axs[3].set_xlabel("Pattern Checks")

        axs[4].set_title(f"Errors (Arc Seconds): Median = {np.nanmedian(self.accuracies):.2f}, Mean = {np.nanmean(self.accuracies):.2f}, Max = {np.nanmax(self.accuracies):.2f}. Min = {np.nanmin(self.accuracies):.2f}, Std Dev = {np.nanstd(self.accuracies):.2f}")
        axs[4].hist(self.accuracies, edgecolor='black', linewidth=1.2, bins=50)
        axs[4].set_xlabel("Accuracy (Arc Seconds)")

        axs[5].set_title("Hash Checks vs Time (seconds)")
        axs[5].scatter(self.hash_checks, self.times)
        axs[5].set_xlabel("Hash Checks")
        axs[5].set_ylabel("Time (Image to Quat)")

        axs[6].set_title("Star Checks vs Pattern Checks")
        axs[6].scatter(self.star_checks, self.pattern_checks)
        axs[6].set_xlabel("Star Checks")
        axs[6].set_ylabel("Pattern Checks")

        plt.tight_layout()
        plt.savefig("runtimes.png")


    def solver_pipeline(self):
        # image_files = sorted(glob(os.path.join(self.stel.image_path, self.args.image_pattern)))
        image_files = glob(os.path.join(self.stel.image_path, self.args.image_pattern))
        self.times = np.zeros(len(image_files)) * np.nan
        self.hash_checks = np.zeros(len(image_files)) * np.nan
        self.star_checks = np.zeros(len(image_files)) * np.nan
        self.pattern_checks = np.zeros(len(image_files)) * np.nan
        self.accuracies = np.zeros(len(image_files)) * np.nan
        self.num_images = len(image_files)
        for i, image_path in enumerate(image_files):
            self.image_number = i
            print(f"Solving for {image_path}")
            self.t1 = time()
            if not self.stel.get_stel_data(image_path):
                continue

            centroid_params = centroiding_pipeline(image_path, args, self.stel)
            star_centroids = centroid_params["star_centroids"]

            self.solve_from_centroids(star_centroids)

        self.output_plots()


if __name__ == "__main__":
    print("Starting....")
    parser = argparse.ArgumentParser(description="Overlay OpenCV blob detection on an image")
    parser.add_argument("--input_path",     default=IMAGE_DIRECTORY,    type=str, help="")
    parser.add_argument("--output_path",    default=OUTPUT_DIRECTORY,   type=str, help="")
    parser.add_argument("--catalog_file",   default=DEFAULT_CATALOG,    type=str, help="")
    parser.add_argument("--coarse_file",    default=DEFAULT_COARSE,     type=str, help="")
    parser.add_argument("--config_path",    default=DEFAULT_CONFIG,     type=str, help="")
    args = parser.parse_args()

    s = StarSolver(args)
    s.solver_pipeline()
