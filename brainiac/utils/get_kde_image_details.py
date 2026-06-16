import os
import argparse
import nibabel as nib


def get_nifti_details(file_path):
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    # 1. Get the physical disk storage size
    disk_size_bytes = os.path.getsize(file_path)
    disk_size_mb = disk_size_bytes / (1024 * 1024)

    # 2. Get the spatial array dimensions from the NIfTI header
    # Using nib.load() is lazy and fast; it doesn't load the raw image voxels into RAM
    img = nib.load(file_path)
    spatial_shape = img.shape
    data_type = img.get_data_dtype()

    print(f"File Details for: {os.path.basename(file_path)}")
    print(f"  Disk Size:        {disk_size_mb:.2f} MB ({disk_size_bytes:,} bytes)")
    print(f"  Matrix Shape:     {spatial_shape}")
    print(f"  Voxel Data Type:  {data_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get the structural shape and disk size of a NIfTI file."
    )
    parser.add_argument(
        "image_path", type=str, help="Path to the target input .nii.gz file"
    )

    args = parser.parse_args()
    get_nifti_details(args.image_path)