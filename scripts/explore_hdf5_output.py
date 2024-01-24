import os
import h5py

# Replace with your HDF5 file path

hdf5_file_path = os.path.join(os.getcwd(), '../results/output.h5')

def explore_hdf5_datasets(file_path):
    with h5py.File(file_path, 'r') as file:
        def print_dataset_info(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset Name: {name}")
                print(f"Dataset Shape: {obj.shape}")
                print(f"Dataset Datatype: {obj.dtype}")

                # Print a small portion of the dataset, up to 20 items
                data_slice = obj[:20] if obj.size > 20 else obj[:]
                print(f"Data Slice (first 20 items or total): {data_slice}")
                print("\n")

        file.visititems(print_dataset_info)

# Run the function with your HDF5 file
explore_hdf5_datasets(hdf5_file_path)
