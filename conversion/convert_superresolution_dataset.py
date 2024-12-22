import os
import numpy as np
from pathlib import Path
from datasets import Dataset, DatasetDict, Features, Value, Image

base_path = Path("/Volumes/SeismicAI/data/SeismicSuperResolution/")
path = base_path / "seismicSuperResolutionData"
lr_path = os.path.join(path, "nx2")
hr_path = os.path.join(path, "sx")

ids = []
lr_images = []
hr_images = []

for file in os.listdir(lr_path):
    if file.endswith(".dat") and len(file.split('.')) == 2:
        file_id = int(file.split('.')[0])
        ids.append(file_id)
        
        lr_file_path = os.path.join(lr_path, file)
        hr_file_path = os.path.join(hr_path, file)
        
        lr_image = np.fromfile(lr_file_path, dtype=np.float32).reshape(128, 128)
        hr_image = np.fromfile(hr_file_path, dtype=np.float32).reshape(256, 256)
        
        lr_images.append(lr_image)
        hr_images.append(hr_image)

# Split data into train, test, and other datasets
train_ids = range(1, 1201)
test_ids = range(1451, 1601)

train_lr_images = [lr_images[i] for i in train_ids]
train_hr_images = [hr_images[i] for i in train_ids]

test_lr_images = [lr_images[i] for i in test_ids]
test_hr_images = [hr_images[i] for i in test_ids]

dset_dict = DatasetDict()

train_features = Features({
    "id": Value("int32"),
    "lr": Image(),
    "hr": Image(),
})

test_features = Features({
    "id": Value("int32"),
    "lr": Image(),
    "hr": Image(),
})

train_dataset = Dataset.from_dict({
    "id": train_ids,
    "lr": train_lr_images,
    "hr": train_hr_images
}, features=train_features)

test_dataset = Dataset.from_dict({
    "id": test_ids,
    "lr": test_lr_images,
    "hr": test_hr_images
}, features=test_features)

dset_dict["train"] = train_dataset
dset_dict["test"] = test_dataset

dset_dict.push_to_hub("porestar/superresolutiondataset", private=True)
