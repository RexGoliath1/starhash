import os
import h5py
import numpy as np
import pandas as pd

# Replace with your HDF5 file path

hdf5_file_path = os.path.join(os.getcwd(), '../results/output.h5')
csv_file_path = os.path.join(os.getcwd(), '../results/proper_motion_data.csv')
def explore_hdf5_datasets(file_path):
    with h5py.File(file_path, 'r') as file:
        def print_dataset_info(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset Name: {name}")
                print(f"Dataset Shape: {obj.shape}")
                print(f"Dataset Datatype: {obj.dtype}")

        file.visititems(print_dataset_info)

# Run the function with your HDF5 file
test = explore_hdf5_datasets(hdf5_file_path)

with h5py.File(hdf5_file_path, 'r') as hf:
    catalog_data = np.array(hf.get('input_catalog_data'))

df = pd.read_csv(csv_file_path, header=None)
df.columns =["x","y","z"]
df["HIP"] = catalog_data[:,2]
df2 = pd.DataFrame(catalog_data)
df2.columns = "RA_J2000, DE_J2000, HID, RA_ICRS, DE_ICRS, PLX, PMRA, PMDE, HPMAG, COLOUR".split(", ")
print(df2.head)
