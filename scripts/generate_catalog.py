import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import itertools
import yaml
import argparse

default_input = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "hipparcos.csv")
default_output= os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
default_scg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

_MAGIC_RAND = 2654435761

class tetra():
    """
        ---------------------------------- VALIDATION ONLY ----------------------------------
        A copy of the original ESA Tetra baseline to spotcheck StarCatalogGenerator.
        Tests that coarse sky map, star_table, and pattern_catalog are 1:1.
        Significantly slower in python due to list / array appending. Not intended to produce FSW Catalog.
        ---------------------------------- VALIDATION ONLY ----------------------------------
    """
    def __init__(self, args):
        self.output_path            = args.output_path
        self.input_catalog          = args.input_catalog
        self.max_fov                = 20.0
        self.pattern_stars_per_fov  = 5 # 10
        self.catalog_stars_per_fov  = 10 # 20
        self.star_min_separation    = 0.05
        self.pattern_max_error      = 0.005
        self.pattern_size           = 4
        self.pattern_bins           = 25
        self.min_brightness         = 6.0
        self.scg_path               = args.scg_path
        self.coarse_only            = args.coarse_only

        self.max_fov = np.deg2rad(float(self.max_fov))

        self.filter_input_cat()
        self.verify_star_table()

    def _key_to_index(self, key, bin_factor, max_index):
        """Get hash index for a given key."""
        # Get key as a single integer
        index = sum(int(val) * int(bin_factor)**i for (i, val) in enumerate(key))
        # Randomise by magic constant and modulo to maximum index
        return (index * _MAGIC_RAND) % max_index

    def filter_input_cat(self):
        star_table = pd.read_csv(self.input_catalog)
        idx = star_table["Hpmag"] < self.min_brightness
        star_table = star_table[idx]
        idx = star_table["Plx"] > 0.0
        star_table = star_table[idx]

        ra  = np.deg2rad(star_table["RAJ2000"])
        dec = np.deg2rad(star_table["DEJ2000"])
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
        keep_for_patterns[np.int32(0)] = True
        keep_for_verifying[np.int32(0)] = True
        for star_ind in tqdm(range(1, num_stars)):
            star_ind = np.uint32(star_ind)
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

            # `np.int` was a deprecated alias for the builtin `int`. 
            # To avoid this error in existing code, use `int` by itself.
            hash_code = tuple(((vector+1)*temp_bins).astype(int))
            # if the partition is empty, create a new list to hold the star
            # if the partition already contains stars, add the star to the list
            temp_coarse_sky_map[hash_code] = temp_coarse_sky_map.pop(hash_code, []) + [star_id]

        def np_int32_to_int(value):
            if isinstance(value, np.int32):
                return int(value)
            if isinstance(value, np.int64):
                # This is not safe for star id's > 2**31
                return int(value)
            return value
        
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                # Convert the entire array to a list of ints
                return obj.astype(int).tolist()
            elif isinstance(obj, tuple):
                # Convert tuple to a string representation
                return ', '.join(map(str, obj))
                # Convert tuple elements to ints
                # return tuple(np_int32_to_int(item) for item in obj)
            elif isinstance(obj, list):
                # Convert list elements to ints
                return [np_int32_to_int(item) for item in obj]
            elif isinstance(obj, dict):
                # Apply conversion to both keys and values in the dictionary
                return {convert_to_serializable(key): convert_to_serializable(value) for key, value in obj.items()}
            else:
                return obj
        
        # Register a representer for numpy int32 to convert it to int
        yaml.add_representer(np.int32, lambda dumper, data: dumper.represent_int(int(data)))
        
        # Convert your data structure to a serializable format
        serializable_data = convert_to_serializable(temp_coarse_sky_map)
        
        # Now dump the serializable data to a file
        yml_file = os.path.join(self.output_path, "pycoarse_sky_map.yaml")
        serial_yml_file = os.path.join(self.output_path, "serial_pycoarse_sky_map.yaml")
        with open(yml_file, 'w') as outfile:
            yaml.dump(temp_coarse_sky_map, outfile, default_flow_style=False)
        with open(serial_yml_file, 'w') as outfile:
            yaml.dump(serializable_data, outfile, default_flow_style=False)

        if self.coarse_only:
            return

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

        
        st_file = os.path.join(self.output_path, "py_star_table.csv")
        st = star_table[["x", "y", "z"]]
        st.to_csv(st_file, index=False, header=False)
        pat_file = os.path.join(self.output_path, "py_pattern_catalog.csv")
        pat_df = pd.DataFrame(pattern_catalog)
        pat_df.to_csv(pat_file, index=False, header=False)

        print("Done!")

    def verify_star_table(self):
        """ Validates coarse map, star table, and pattern_catalog """
        py_yml_file = os.path.join(self.output_path, "serial_pycoarse_sky_map.yaml")
        # Load the python version of coarse sky map
        with open(py_yml_file, 'r') as file:
            pydata = yaml.safe_load(file)

        # Load the cpp version of coarse sky map
        scg_yml_file = os.path.join(self.scg_path, "coarse_sky_map.yaml")
        with open(scg_yml_file, 'r') as file:
            scgdata = yaml.safe_load(file)

        s1 = set(pydata.keys())
        s2 = set(scgdata.keys())
        pydiff = s1.difference(s2)
        scgdiff = s2.difference(s1)
        print(f"Number of sets in python different: {len(pydiff)}")
        print(f"Number of sets in SCG different: {len(scgdiff)}")

        assert(set(scgdata.keys()) == set(pydata.keys()))

        for key in scgdata.key():
            assert(scgdata[key] == pydata[key])

if __name__ == "__main__":
    def valid_path(s):
        if os.path.exists(s):
            return s
        else:
            raise NotADirectoryError(s)

    parser = argparse.ArgumentParser(description="Python catalog generation validation code. Used to check against Star Catalog Generator ")
    parser.add_argument("--input_catalog", default=default_input, type=valid_path, help="")
    parser.add_argument("--output_path", default=default_output, type=valid_path, help="")
    parser.add_argument("--scg_path", default=default_scg_path, type=valid_path, help="")
    parser.add_argument("--coarse_only", default=True, type=bool, help="")
    args = parser.parse_args()

    cat = tetra(args)
