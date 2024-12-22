import os
import numpy as np
from datasets import Dataset, DatasetDict, Features, Value, Image
from pathlib import Path

def ensure_dir_exists(path: str):
    """Ensure that a directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)

def load_binary_image(filepath: str, dtype: np.dtype, shape: tuple) -> bytes:
    """Load a binary image file as a numpy array and return it as bytes."""
    arr = np.fromfile(filepath, dtype=dtype).reshape(shape)
    return arr

def gather_image_data(file_ids: list, base_dir: str, dtype: np.dtype, shape: tuple) -> list:
    """Load binary image data for a given list of file IDs from a directory."""
    images = []
    for f_id in file_ids:
        filepath = os.path.join(base_dir, f"{f_id}.dat")
        images.append(load_binary_image(filepath, dtype, shape))
    return images

def write_dataset_from_dirs(base_path: str,
                            output_dir: str,
                            image_shape: tuple,
                            seismic_dtype: np.dtype = np.float32,
                            label_dtype: np.dtype = np.float32,
                            push_to_hub_name: str = None,
                            private: bool = True, 
                            directory_names: list = ["image", "label"]):
    """
    Create a parquet dataset from a directory structure of the form:
    base_path/{split}/seismic/*.dat
    base_path/{split}/label/*.dat

    Where {split} can be 'train', 'validation', 'test' (or any subset of these).

    Only the splits that exist will be processed. It will generate train.parquet,
    validation.parquet, and/or test.parquet depending on available splits.

    After creation, it loads the dataset and can optionally push it to the Hugging Face Hub.
    """
    ensure_dir_exists(output_dir)

    # Potential splits to check
    possible_splits = ["train", "valid", "test"]
    dset = DatasetDict()
    for split in possible_splits:
        seismic_dir = os.path.join(base_path, split, directory_names[0])
        label_dir = os.path.join(base_path, split, directory_names[1])

        # Check if both seismic and label directories for this split exist
        if not (os.path.isdir(seismic_dir) and os.path.isdir(label_dir)):
            continue

        # Gather file IDs by listing .dat files in the seismic directory
        file_ids = []
        for f in os.listdir(seismic_dir):
            if f.endswith(".dat"):
                # Extract file ID from filename (e.g., "0.dat" -> 0)
                try:
                    file_id = int(os.path.splitext(f)[0])
                    file_ids.append(file_id)
                except ValueError:
                    pass

        file_ids.sort()

        if not file_ids:
            # No files for this split
            continue

        # Gather data for this split
        seismic_images = gather_image_data(file_ids, seismic_dir, seismic_dtype, image_shape)
        label_images = gather_image_data(file_ids, label_dir, label_dtype, image_shape)
        label_images = [img.astype(np.uint8) for img in label_images]

        columns = {
            'id': file_ids,
            'seismic': seismic_images,
            'label': label_images,
        }

        features = Features({
            "id": Value("int32"),
            "seismic": Image(),
            "label": Image(),
        })
        dataset_split = Dataset.from_dict(columns, features=features)
        dataset_split.set_format(type="torch")
        dset[split] = dataset_split

    # Load all splits into a dataset dictionary
    if push_to_hub_name is not None:
        dset.push_to_hub(push_to_hub_name, private=private)
    return dset

########################################
# Example Usage: Facies Dataset (Crater)
########################################

# Suppose we have directories:
# crater/train/seismic, crater/train/label
# crater/validation/seismic, crater/validation/label
# crater/test/seismic, crater/test/label
#
# Each containing files like 0.dat, 1.dat, ...
#
# The code below will automatically detect these splits and create the respective parquet files.

base_path = Path("/Volumes/SeismicAI/data/CrossDomainFoundationModelAdaption/")
crater_path = base_path / "crater"
write_dataset_from_dirs(
    base_path=crater_path,
    output_dir="data/crossdomainfoundationmodeladaption/crater",
    image_shape=(1022,1022),
    seismic_dtype=np.float32,
    label_dtype=np.int8,
    push_to_hub_name="porestar/crossdomainfoundationmodeladaption-crater",
    private=True,
    directory_names=["image", "label"]
)


das_path = base_path / "das"
write_dataset_from_dirs(
    base_path=das_path,
    output_dir="data/crossdomainfoundationmodeladaption/das",
    image_shape=(512, 512),
    seismic_dtype=np.float32,
    label_dtype=np.int8,
    push_to_hub_name="porestar/crossdomainfoundationmodeladaption-das",
    private=True,
    directory_names=["image", "label"]
)


geobody_path = base_path / "geobody"
write_dataset_from_dirs(
    base_path=geobody_path,
    output_dir="data/crossdomainfoundationmodeladaption/geobody",
    image_shape=(224, 224),
    seismic_dtype=np.float32,
    label_dtype=np.int8,
    push_to_hub_name="porestar/crossdomainfoundationmodeladaption-geobody",
    private=True,
    directory_names=["input", "target"]
)


seismic_path = base_path / "seismicFace"
write_dataset_from_dirs(
    base_path=seismic_path,
    output_dir="data/crossdomainfoundationmodeladaption/seismicfacies",
    image_shape=(1006,782),
    seismic_dtype=np.float32,
    label_dtype=np.int8,
    push_to_hub_name="porestar/crossdomainfoundationmodeladaption-seismicfacies",
    private=True,
    directory_names=["input", "target"]
)

fault_path = base_path / "deepfault"
write_dataset_from_dirs(
    base_path=fault_path,
    output_dir="data/crossdomainfoundationmodeladaption/deepfault",
    image_shape=(896,896),
    seismic_dtype=np.float32,
    label_dtype=np.int8,
    push_to_hub_name="porestar/crossdomainfoundationmodeladaption-deepfault",
    private=True,
    directory_names=["image", "label"]
)

print("All datasets created and pushed to the Hub (if specified).")