import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import itertools
import yaml
import argparse

default_input = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "hipparcos.csv")
default_output= os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "pyvalid")
default_scg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "scg")

_MAGIC_RAND = 2654435761

def _key_to_index(key, bin_factor, max_index):
    """Get hash index for a given key."""
    # Get key as a single integer
    # index = sum(int(val) * int(bin_factor)**i for (i, val) in enumerate(key))
    index = list(int(val) * int(bin_factor)**i for (i, val) in enumerate(key))
    index_sum = sum(index)
    # Randomise by magic constant and modulo to maximum index
    # print(f"index = {index}")
    return (index_sum * _MAGIC_RAND) % max_index

def compare_df(path1, path2, epsilon = 1**-9):
    df1 = pd.read_csv(path1, header=None)
    df2 = pd.read_csv(path2, header=None)

    assert(df1.shape == df2.shape)
    print(f'Shapes equal: {df1.shape} == {df2.shape}')

    df_diff = df1 - df2
    count = 0
    for ii in range(df_diff.shape[0]):
        row = df_diff.iloc[ii]
        if sum(row) > epsilon:
            count += 1
            print(f"Row {ii} difference")
            print(f"df1 row: {df1.iloc[ii].values.transpose()}")
            print(f"df2 row: {df2.iloc[ii].values.transpose()}")
            exit(-1)
    return count


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
        self.validate_only          = args.validate_only

        self.max_fov = np.deg2rad(float(self.max_fov))

        os.makedirs(self.output_path, exist_ok=True)

        if not self.validate_only:
            self.filter_input_cat()

        self.verify_star_table()


    def filter_input_cat(self):
        star_table = pd.read_csv(self.input_catalog)
        idx = star_table["Hpmag"] < self.min_brightness
        star_table = star_table[idx]
        idx = star_table["Plx"] > 0.0
        star_table = star_table[idx]

        # WARNING: kind='mergesort' is the only stable sort in numpy
        star_table = star_table.sort_values(by=["Hpmag", "HIP"], ascending=[True, True], kind='mergesort')

        sorted_st_file = os.path.join(self.output_path, "py_input_catalog.csv")
        star_table.to_csv(sorted_st_file, index=False, header=False)

        ra  = np.deg2rad(star_table["RAJ2000"])
        dec = np.deg2rad(star_table["DEJ2000"])
        star_table["x"] = np.cos(ra) * np.cos(dec)
        star_table["y"] = np.sin(ra) * np.cos(dec)
        star_table["z"] = np.sin(dec)

        # Filter for maximum number of stars in FOV and doubles
        num_stars = star_table.shape[0]
        keep_for_patterns = [False] * num_stars
        keep_for_verifying = [False] * num_stars
        all_star_vectors = star_table[["x", "y", "z"]].transpose()

        print("Filtering star separation")

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
        verify_stars = star_table.shape[0]
        pattern_stars = (np.cumsum(keep_for_verifying)-1)[keep_for_patterns]

        st_file = os.path.join(self.output_path, "py_star_table.csv")
        pat_st_file = os.path.join(self.output_path, "py_pat_star_table.csv")
        st = star_table[["x", "y", "z"]]
        st.to_csv(st_file, index=False, header=False)
        pat_st= st.iloc[pattern_stars, :]
        pat_st.to_csv(pat_st_file, index=False, header=False)

        print(f"With maximum {self.pattern_stars_per_fov} per FOV and no doubles: {len(pattern_stars)}")
        print(f"With maximum {self.catalog_stars_per_fov} per FOV and no doubles: {verify_stars}")
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
        pattern_list = np.zeros((10000, self.pattern_size))
        pl_count = 0

        # initialize pattern, which will contain pattern_size star ids
        pattern = [None] * self.pattern_size
        for pattern[0] in tqdm(pattern_stars):  # star_ids_filtered:
            vector = star_table[["x", "y", "z"]].iloc[pattern[0]]
            # find which partition the star occupies in the sky hash table
            hash_code = tuple(((vector+1)*temp_bins).astype(np.int32))
            # remove the star from the sky hash table
            temp_coarse_sky_map[hash_code].remove(pattern[0])
            # iterate over all possible patterns containing the removed star
            nearby_ids = temp_get_nearby_stars(vector, self.max_fov)

            # print(f"Looking for patterns near star id {pattern[0]} Number of Neighbors = {len(nearby_ids)}")
            # for pattern[1:] in itertools.combinations(temp_get_nearby_stars(vector, self.max_fov), self.pattern_size-1):
            for pattern[1:] in itertools.combinations(nearby_ids, self.pattern_size-1):
                # retrieve the vectors of the stars in the pattern
                vectors = star_table[["x", "y", "z"]].iloc[pattern].values
                patarray = np.array(pattern)
                # verify that the pattern fits within the maximum field-of-view
                # by checking the distances between every pair of stars in the pattern
                if all(np.dot(*star_pair) > np.cos(self.max_fov) for star_pair in itertools.combinations(vectors, 2)):
                    # print(f"pattern = {pattern}")
                    pattern_list[pl_count, :] = patarray

                    pl_count += 1
                    if pl_count >= pattern_list.shape[0]:
                        newshape = (pattern_list.shape[0] + 10000, pattern_list.shape[1])
                        pattern_list.resize(newshape, refcheck=False)

        pattern_list = pattern_list[:pl_count, :]
        pl_file = os.path.join(self.output_path, "py_pattern_list.csv")
        pl_df = pd.DataFrame(pattern_list)
        pl_df.to_csv(pl_file, index=False, header=False)

        print(f"Found {pattern_list.shape[0]} patterns. Building catalogue.")
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
            hash_index = _key_to_index(hash_code, self.pattern_bins, catalog_length)
            # print(f"pattern[{hash_index}].hash_code = {hash_code} (edges: {edge_ratios})")
            # use quadratic probing to find an open space in the pattern catalog to insert
            for index in ((hash_index + offset ** 2) % catalog_length
                          for offset in itertools.count()):
                # if the current slot is empty, add the pattern
                # possibly wrong?
                # if not pattern_catalog[index].sum() > 0:
                if not pattern_catalog[index].sum() > 0:
                    pattern_catalog[index] = pattern
                    break
        
        pat_file = os.path.join(self.output_path, "py_pattern_catalog.csv")
        pat_df = pd.DataFrame(pattern_catalog)
        pat_df.to_csv(pat_file, index=False, header=False)

        print("Done!")

    def verify_star_table(self):
        """ Validates input catalog, coarse map, star table, and pattern_catalog """

        # Validate Input Catalog first
        py_ic = os.path.join(self.output_path, "py_input_catalog.csv")
        scg_ic = os.path.join(self.scg_path, "input_catalog.csv")
        count = compare_df(py_ic, scg_ic)
        print(f"Input Catalog: Done checking row difference. Total Differences: {count}")

        # Validate Star Table second
        py_stf = os.path.join(self.output_path, "py_star_table.csv")
        scg_stf = os.path.join(self.scg_path, "star_table_ut_no_motion.csv")
        count = compare_df(py_stf, scg_stf)
        print(f"Validation Star Table: Done checking row difference. Total Differences: {count}")

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

        pydiffcnt = 0
        scgdiffcnt = 0
        totaldiffcnt = 0
        for key in scgdata.keys():
            totaldiffcnt += 1
            ss1 = set(pydata[key])
            ss2 = set(scgdata[key])
            ss1diff = ss1.difference(ss2)
            ss2diff = ss2.difference(ss1)
            if (len(ss1diff) > 0):
                print(f"pyt[{key}].diff = {ss1diff}")
                pydiffcnt += 1
            if (len(ss2diff) > 0):
                print(f"scg[{key}].diff = {ss2diff}")
                scgdiffcnt += 1

        print(f"Python number of coarse key indicies different : {pydiffcnt} / {totaldiffcnt}")
        print(f"SCG number of coarse key indicies different : {scgdiffcnt} / {totaldiffcnt}")

        # Validate Pattern List
        py_pl = os.path.join(self.output_path, "py_pattern_list.csv")
        scg_pl = os.path.join(self.scg_path, "pattern_list_ut_no_motion.csv")
        count = compare_df(py_pl, scg_pl)
        print(f"Validation Pattern List: Done checking row difference. Total Differences: {count}")

        # Validate Pattern Catalog
        py_pc = os.path.join(self.output_path, "py_pattern_catalog.csv")
        scg_pc = os.path.join(self.scg_path, "pattern_catalog_ut_no_motion.csv")
        count = compare_df(py_pc, scg_pc)
        print(f"Validation Pattern Catalog: Done checking row difference. Total Differences: {count}")


        self.index_all(py_stf, py_pc)

        print("Done!")

    def index_all(self, stf, py_pc):
        """ Load star table file and index into pattern catalog. Check every pattern

        """

        star_table = pd.read_csv(stf, header=None)
        star_table.columns = ["x", "y", "z"]
        pattern_catalog = pd.read_csv(py_pc, header=None)
        pattern_catalog = pattern_catalog.to_numpy()
        catalog_length = pattern_catalog.shape[0]

        for ii in tqdm(range(pattern_catalog.shape[0])):
            # For each row in the pattern catalog, let's find the actual pattern from star table
            pattern = pattern_catalog[ii]

            if pattern.sum() == 0:
                continue
            if any(pattern < 0):
                raise Exception("Negative values in pattern catalog")

            vectors = star_table[["x", "y", "z"]].iloc[pattern].values
            # calculate and sort the edges of the star pattern
            edges = np.sort([np.sqrt((np.subtract(*star_pair)**2).sum())
                             for star_pair in itertools.combinations(vectors, 2)])
            # extract the largest edge
            largest_edge = edges[-1]
            # divide the edges by the largest edge to create dimensionless ratios
            edge_ratios = edges[:-1] / largest_edge

            hash_code = tuple((edge_ratios * self.pattern_bins).astype(np.int32))
            hash_index = _key_to_index(hash_code, self.pattern_bins, catalog_length)
            assert(hash_index > 0)
            match = False

            for index in ((hash_index + offset ** 2) % catalog_length
                          for offset in itertools.count()):
                # Proceed with quadratic probing until we hit an empty spot
                # TODO: WARNING, POSSIBLE BAD CHECK IN PYTHON AND C++, MAY NEED DEBUGGING
                if pattern_catalog[index].sum() == 0:
                    break
                if index == ii:
                    match = True

            assert(match)

if __name__ == "__main__":
    def valid_path(s):
        if os.path.exists(s):
            return s
        else:
            raise NotADirectoryError(s)

    parser = argparse.ArgumentParser(description="Python catalog generation validation code. Used to check against Star Catalog Generator ")
    parser.add_argument("--input_catalog", default=default_input, type=valid_path, help="")
    parser.add_argument("--output_path", default=default_output, type=str, help="")
    parser.add_argument("--scg_path", default=default_scg_path, type=valid_path, help="")
    parser.add_argument("--coarse_only", default=False, type=bool, help="")
    parser.add_argument("--validate_only", default=True, type=bool, help="Validate without rerunning python catalog generation")
    args = parser.parse_args()
    settings = vars(args)

    cat = tetra(args)

