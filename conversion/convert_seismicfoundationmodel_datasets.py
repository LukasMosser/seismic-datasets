import os
import numpy as np
from pathlib import Path
from datasets import Dataset, DatasetDict, Features, Value, Image
from seismic_datasets.utils import ensure_dir_exists, gather_image_data

def write_dataset(split_file_ids: dict,
                  seismic_dir: str,
                  label_dir: str,
                  output_dir: str,
                  image_shape: tuple,
                  seismic_dtype: np.dtype = np.float32,
                  label_dtype: np.dtype = np.float32,
                  push_to_hub_name: str = None,
                  private: bool = True):
    """
    Create a parquet dataset from seismic and label directories for multiple splits
    and push it to the Hugging Face Hub if desired.
    
    split_file_ids should be a dictionary like:
    {
      "train": [0,1,2,...],
      "validation": [101,102,...],
      "test": [201,202,...]
    }
    """
    ensure_dir_exists(output_dir)
    dset = DatasetDict()
    # For each split, create a parquet file
    for split_name, file_ids in split_file_ids.items():
        if not file_ids:
            continue

        # Gather data for this split
        seismic_images = gather_image_data(file_ids, seismic_dir, seismic_dtype, image_shape)
        label_images = gather_image_data(file_ids, label_dir, label_dtype, image_shape)

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
        dset[split_name] = dataset_split

    if push_to_hub_name is not None:
        dset.push_to_hub(push_to_hub_name, private=private)
    return dset

def write_single_modal_dataset(split_file_ids: dict,
                               data_dir: str,
                               output_dir: str,
                               image_shape: tuple,
                               dtype: np.dtype = np.float32,
                               push_to_hub_name: str = None,
                               private: bool = True):
    """
    Create a parquet dataset with a single binary column for multiple splits.
    
    split_file_ids should be a dictionary like:
    {
      "train": [0,1,2,...],
      "validation": [101,102,...],
      "test": [201,202,...]
    }
    """
    ensure_dir_exists(output_dir)
    dset = DatasetDict()
    for split_name, file_ids in split_file_ids.items():
        if not file_ids:
            continue

        images = gather_image_data(file_ids, data_dir, dtype, image_shape)

        columns = {
            'id': file_ids,
            'seismic': images,
        }

        features = Features({
            "id": Value("int32"),
            "seismic": Image(),
        })

        dataset_split = Dataset.from_dict(columns, features=features)
        dataset_split.set_format(type="torch")
        dset[split_name] = dataset_split


    if push_to_hub_name is not None:
        dset.push_to_hub(push_to_hub_name, private=private)
    return dset

##################################
# Example usage with the new format
##################################

### Denoise Dataset ###

base_path = "/Volumes/SeismicAI/data/SeismicFoundationModel/Denoise"
denoise_ids = []
for file in os.listdir(os.path.join(base_path, "seismic")):
    if file.endswith(".dat") and len(file.split('.')) == 2:
        denoise_ids.append(int(file.split('.')[0]))

# If you only have a single split (train), just put them all under 'train'.
denoise_splits = {"train": denoise_ids}

write_dataset(
    split_file_ids=denoise_splits,
    seismic_dir=os.path.join(base_path, "seismic"),
    label_dir=os.path.join(base_path, "label"),
    output_dir="data/seismicfoundationmodel/denoise",
    image_shape=(224,224),
    push_to_hub_name="porestar/seismicfoundationmodel-denoise",
    private=True
)

### Denoise-field Dataset ###

denoise_field_ids = []
for file in os.listdir(os.path.join(base_path, "field")):
    if file.endswith(".dat") and len(file.split('.')) == 2:
        denoise_field_ids.append(int(file.split('.')[0]))

denoise_field_splits = {"train": denoise_field_ids}

write_single_modal_dataset(
    split_file_ids=denoise_field_splits,
    data_dir=os.path.join(base_path, "field"),
    output_dir="data/seismicfoundationmodel/denoise-field",
    image_shape=(224,224),
    push_to_hub_name="porestar/seismicfoundationmodel-denoise-field",
    private=True
)

base_path = Path("/Volumes/SeismicAI/data/SeismicFoundationModel")

### Facies Dataset ###
# 0 - 100 train, 101 - 117 validation
facies_train_ids = list(range(0, 100))
facies_val_ids = list(range(100, 117))
facies_splits = {"train": facies_train_ids, "validation": facies_val_ids}
facies_path = base_path / "Facies"

write_dataset(
    split_file_ids=facies_splits,
    seismic_dir=os.path.join(facies_path, "seismic"),
    label_dir=os.path.join(facies_path, "label"),
    output_dir="data/seismicfoundationmodel/facies",
    image_shape=(768,768),
    seismic_dtype=np.float32,
    label_dtype=np.int32,
    push_to_hub_name="porestar/seismicfoundationmodel-facies",
    private=True
)


### Geobody Dataset ###

# 0 - 3500 train, 3501 - 4000 validation
geobody_train_ids = list(range(0,3500))
geobody_val_ids = list(range(3500,4000))
geobody_splits = {"train": geobody_train_ids, "validation": geobody_val_ids}
geobody_path = base_path / "Geobody"

write_dataset(
    split_file_ids=geobody_splits,
    seismic_dir=os.path.join(geobody_path, "seismic"),
    label_dir=os.path.join(geobody_path, "label"),
    output_dir="data/seismicfoundationmodel/geobody",
    image_shape=(224,224),
    seismic_dtype=np.float32,
    label_dtype=np.int32,
    push_to_hub_name="porestar/seismicfoundationmodel-geobody",
    private=True
)


### Interpolation Dataset ###

# 0 - 3500 train, 3501 - 4000 validation (adjust as needed)
interpolation_train_ids = list(range(0,3500))
interpolation_val_ids = list(range(3500,4000))
interpolation_splits = {"train": interpolation_train_ids, "validation": interpolation_val_ids}
interpolation_path = base_path / "Interpolation"

# Assuming seismic_dir and label_dir are the same path for interpolation as per original code
write_single_modal_dataset(
    split_file_ids=interpolation_splits,
    data_dir=interpolation_path,
    output_dir="data/seismicfoundationmodel/interpolation",
    image_shape=(224,224),
    push_to_hub_name="porestar/seismicfoundationmodel-interpolation",
    private=True
)

### Inversion Dataset ###

# 0 - 2199 training, 5000 validation (example IDs, adjust as needed)
inversion_train_ids = list(range(0,2200))
inversion_val_ids = []
inversion_splits = {"train": inversion_train_ids, "validation": inversion_val_ids}
inversion_path = base_path / "Inversion"

write_dataset(
    split_file_ids=inversion_splits,
    seismic_dir=os.path.join(inversion_path, "seismic"),
    label_dir=os.path.join(inversion_path, "label"),
    output_dir="data/seismicfoundationmodel/inversion-synthetic",
    image_shape=(224,224),
    seismic_dtype=np.float32,
    label_dtype=np.float32,
    push_to_hub_name="porestar/seismicfoundationmodel-inversion-synthetic",
    private=True
)

# 0 - 2199 training, 5000 validation (example IDs, adjust as needed)
inversion_train_ids = list(range(0,5000))
inversion_val_ids = []
inversion_splits = {"train": inversion_train_ids, "validation": inversion_val_ids}
inversion_path = base_path / "Inversion"

write_dataset(
    split_file_ids=inversion_splits,
    seismic_dir=os.path.join(inversion_path, "SEAMseismic"),
    label_dir=os.path.join(inversion_path, "SEAMreflect"),
    output_dir="data/seismicfoundationmodel/inversion-seam",
    image_shape=(224,224),
    seismic_dtype=np.float32,
    label_dtype=np.float32,
    push_to_hub_name="porestar/seismicfoundationmodel-inversion-seam",
    private=True
)

print("All datasets created and (optionally) pushed to the Hub.")