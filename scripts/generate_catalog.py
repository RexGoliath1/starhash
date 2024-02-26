import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import itertools

default_input = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "hipparcos.csv")
default_output = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "catalog.hdf5")

_MAGIC_RAND = 2654435761

class tetra():
    def __init__(self, input_catalog: str = default_input, output_file: str = default_output):
        self.output_db              = output_file
        self.max_fov                = 20.0
        self.pattern_stars_per_fov  = 10
        self.catalog_stars_per_fov  = 20
        self.star_min_separation    = 0.05
        self.pattern_max_error      = 0.005
        self.pattern_size           = 4
        self.pattern_bins           = 25
        self.min_brightness         = 6.0

        self.max_fov = np.deg2rad(float(self.max_fov))
        self.filter_input_cat(input_catalog)

    def _key_to_index(self, key, bin_factor, max_index):
        """Get hash index for a given key."""
        # Get key as a single integer
        index = sum(int(val) * int(bin_factor)**i for (i, val) in enumerate(key))
        # Randomise by magic constant and modulo to maximum index
        return (index * _MAGIC_RAND) % max_index

    def filter_input_cat(self, input_catalog: str):
        star_table = pd.read_csv(input_catalog)
        idx = star_table["Hpmag"] < self.min_brightness
        star_table = star_table[idx]

        ra  = np.deg2rad(star_table["_RAICRS"])
        dec = np.deg2rad(star_table["_DEICRS"])
        star_table["x"] = np.cos(ra) * np.cos(dec)
        star_table["y"] = np.sin(ra) * np.cos(dec)
        star_table["z"] = np.sin(dec)
        idx = np.argsort(star_table["Hpmag"])
        star_table = star_table.iloc[idx] 

        # Filter for maximum number of stars in FOV and doubles
        num_stars = star_table.shape[0]
        keep_for_patterns = [False] * num_stars
        keep_for_verifying = [False] * num_stars
        all_star_vectors = star_table[["x", "y", "z"]].transpose()

        # Keep the first one and skip index 0 in loop
        keep_for_patterns[0] = True
        keep_for_verifying[0] = True
        for star_ind in tqdm(range(1, num_stars)):
            vector = star_table[["x", "y", "z"]].iloc[star_ind]
            # Angle to all stars we have kept
            angs_patterns = np.dot(vector, all_star_vectors.loc[:, keep_for_patterns])
            angs_verifying = np.dot(vector, all_star_vectors.loc[:, keep_for_verifying])
            # Check double star limit as well as stars-per-fov limit
            if self.star_min_separation is None \
                    or all(angs_patterns < np.cos(np.deg2rad(self.star_min_separation))):
                num_stars_in_fov = sum(angs_patterns > np.cos(self.max_fov/2))
                if num_stars_in_fov < self.pattern_stars_per_fov:
                    # Only keep if not too many close by already
                    keep_for_patterns[star_ind] = True
                    keep_for_verifying[star_ind] = True
            # Secondary stars-per-fov check, if we fail this we will not keep the star at all
            if self.star_min_separation is None \
                    or all(angs_verifying < np.cos(np.deg2rad(self.star_min_separation))):
                num_stars_in_fov = sum(angs_verifying > np.cos(self.max_fov/2))
                if num_stars_in_fov < self.catalog_stars_per_fov:
                    # Only keep if not too many close by already
                    keep_for_verifying[star_ind] = True
        # Trim down star table and update indexing for pattern stars
        star_table = star_table.loc[keep_for_verifying, :]
        pattern_stars = (np.cumsum(keep_for_verifying)-1)[keep_for_patterns]

        print(f"With maximum {self.pattern_stars_per_fov} per FOV and no doubles: {len(pattern_stars)}")
        print(f"With maximum {self.catalog_stars_per_fov} per FOV and no doubles: {num_stars}")
        print('Building temporary hash table for finding pattern neighbours')

        temp_coarse_sky_map = {}
        temp_bins = 4
        # insert the stars into the hash table
        for star_id in tqdm(pattern_stars):
            vector = star_table[["x", "y", "z"]].iloc[star_id]
            # find which partition the star occupies in the hash table
            hash_code = tuple(((vector+1)*temp_bins).astype(np.int32))
            # if the partition is empty, create a new list to hold the star
            # if the partition already contains stars, add the star to the list
            temp_coarse_sky_map[hash_code] = temp_coarse_sky_map.pop(hash_code, []) + [star_id]

        def temp_get_nearby_stars(vector, radius):
            """Get nearby from temporary hash table."""
            # create list of nearby stars
            nearby_star_ids = []
            # given error of at most radius in each dimension, compute the space of hash codes
            hash_code_space = [range(max(low, 0), min(high+1, 2*temp_bins)) for (low, high)
                               in zip(((vector + 1 - radius) * temp_bins).astype(np.int32),
                                      ((vector + 1 + radius) * temp_bins).astype(np.int32))]
            # iterate over hash code space
            for hash_code in itertools.product(*hash_code_space):
                # iterate over the stars in the given partition, adding them to
                # the nearby stars list if they're within range of the vector
                for star_id in temp_coarse_sky_map.get(hash_code, []):
                    if np.dot(vector, star_table[["x", "y", "z"]].iloc[star_id]) > np.cos(radius):
                        nearby_star_ids.append(star_id)
            return nearby_star_ids

        # generate pattern catalog
        print('Generating all possible patterns.')
        pattern_list = []
        # initialize pattern, which will contain pattern_size star ids
        pattern = [None] * self.pattern_size
        for pattern[0] in tqdm(pattern_stars):  # star_ids_filtered:
            vector = star_table[["x", "y", "z"]].iloc[pattern[0]]
            # find which partition the star occupies in the sky hash table
            hash_code = tuple(((vector+1)*temp_bins).astype(np.int32))
            # remove the star from the sky hash table
            temp_coarse_sky_map[hash_code].remove(pattern[0])
            # iterate over all possible patterns containing the removed star
            for pattern[1:] in itertools.combinations(temp_get_nearby_stars(vector, self.max_fov), self.pattern_size-1):
                # retrieve the vectors of the stars in the pattern
                vectors = star_table[["x", "y", "z"]].iloc[pattern].values
                # verify that the pattern fits within the maximum field-of-view
                # by checking the distances between every pair of stars in the pattern
                # all_good = True
                # for star_pair in itertools.combinations(vectors, 2):
                #     if not np.dot(*star_pair) > np.cos(self.max_fov):
                #         all_good = False

                # if all_good:
                #     pattern_list.append(pattern.copy())

                if all(np.dot(*star_pair) > np.cos(self.max_fov) for star_pair in itertools.combinations(vectors, 2)):
                    pattern_list.append(pattern.copy())


        print(f"Found {len(pattern_list)} patterns. Building catalogue.")
        catalog_length = 2 * len(pattern_list)
        pattern_catalog = np.zeros((catalog_length, self.pattern_size), dtype=np.uint16)
        for pattern in tqdm(pattern_list):
            # retrieve the vectors of the stars in the pattern
            vectors = star_table[["x", "y", "z"]].iloc[pattern].values
            # calculate and sort the edges of the star pattern
            edges = np.sort([np.sqrt((np.subtract(*star_pair)**2).sum())
                             for star_pair in itertools.combinations(vectors, 2)])
            # extract the largest edge
            largest_edge = edges[-1]
            # divide the edges by the largest edge to create dimensionless ratios
            edge_ratios = edges[:-1] / largest_edge
            # convert edge ratio float to hash code by binning
            hash_code = tuple((edge_ratios * self.pattern_bins).astype(np.int32))
            hash_index = self._key_to_index(hash_code, self.pattern_bins, catalog_length)
            # use quadratic probing to find an open space in the pattern catalog to insert
            for index in ((hash_index + offset ** 2) % catalog_length
                          for offset in itertools.count()):
                # if the current slot is empty, add the pattern
                if not pattern_catalog[index][0]:
                    pattern_catalog[index] = pattern
                    break

        print("Done!")


if __name__ == "__main__":
    cat = tetra()
