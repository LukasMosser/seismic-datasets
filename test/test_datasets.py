# test_datasets.py
import pytest
from datasets import load_dataset
import numpy as np 

#
# Cross-Domain Foundation Model Adaptation
#

@pytest.mark.parametrize("dataset_name,expected_splits,expected_shapes", [
    # Each entry: (HF repo name, [list of splits], {"split_name": shape_of_seismic, ...})
    (
        "porestar/crossdomainfoundationmodeladaption-crater",
        ["train", "valid"], 
        {
            "train": (1022, 1022), 
            "valid": (1022, 1022)
        }
    ),
    (
        "porestar/crossdomainfoundationmodeladaption-das",
        ["train", "valid"], 
        {
            "train": (512, 512), 
            "valid": (512, 512)
        }
    ),
    (
        "porestar/crossdomainfoundationmodeladaption-geobody",
        ["train", "valid"], 
        {
            "train": (224, 224), 
            "valid": (224, 224)
        }
    ),
    (
        "porestar/crossdomainfoundationmodeladaption-seismicfacies",
        ["train", "valid"], 
        {
            "train": (1006, 782), 
            "valid": (1006, 782)
        }
    ),
    (
        "porestar/crossdomainfoundationmodeladaption-deepfault",
        ["train", "valid"], 
        {
            "train": (896, 896), 
            "valid": (896, 896)
        }
    ),
])
def test_crossdomainfoundationmodeladaption_datasets(dataset_name, expected_splits, expected_shapes):
    dataset = load_dataset(dataset_name).with_format("numpy")

    # Check that the dataset has the expected splits
    actual_splits = list(dataset.keys())
    for split in expected_splits:
        assert split in actual_splits, f"Expected split '{split}' not found in dataset {dataset_name}. Found splits: {actual_splits}"

    # Check shape of "seismic" and "label" if it is a multi-modal dataset
    # or just "seismic" if single-modal. Many of these have both "seismic" and "label".
    for split in expected_splits:
        if split not in dataset:
            continue
        split_dataset = dataset[split]
        # If there's a "seismic" column, test shape
        if "seismic" in split_dataset.features:
            item = split_dataset.select([0])["seismic"].squeeze()
            seismic_shape = item.shape
            # e.g. (1022, 1022)
            assert seismic_shape == expected_shapes[split], (
                f"Expected seismic shape {expected_shapes[split]} in split '{split}' for dataset {dataset_name}, "
                f"but got {seismic_shape}."
            )
        # If there's a "label" column, test shape as well
        if "label" in split_dataset.features:
            item = split_dataset.select([0])["label"].squeeze()
            label_shape = item.shape
            # Usually label is the same shape as seismic, but sometimes different integer dtype
            # If the code above created them at the same shape, we can do:
            assert label_shape == expected_shapes[split], (
                f"Expected label shape {expected_shapes[split]} in split '{split}' for dataset {dataset_name}, "
                f"but got {label_shape}."
            )

#
# Seismic Foundation Model Datasets
#

@pytest.mark.parametrize("dataset_name, expected_splits, expected_shapes, expected_sizes", [
    #
    # Denoise
    #
    (
        "porestar/seismicfoundationmodel-denoise",
        ["train"],  # Only one split
        {"train": (224, 224)},
        {"train": 2000}  # from your original snippet
    ),
    #
    # Denoise-field (single-modal, "seismic" only)
    #
    (
        "porestar/seismicfoundationmodel-denoise-field",
        ["train"],
        {"train": (224, 224)},
        {"train": None}  # If you don't know the count, use None or skip
    ),
    #
    # Facies dataset 
    #
    (
        "porestar/seismicfoundationmodel-facies",
        ["train", "validation"], 
        {"train": (768, 768), "validation": (768, 768)},
        {"train": 100, "validation": 17}  # from your code snippet (0-100 => 100 items, 100-117 => 17 items)
    ),
    #
    # Geobody dataset
    #
    (
        "porestar/seismicfoundationmodel-geobody",
        ["train", "validation"], 
        {"train": (224, 224), "validation": (224, 224)},
        {"train": 3500, "validation": 500}  # from your code snippet (0-3500 => 3500 items, 3500-4000 => 500 items)
    ),
    #
    # Interpolation dataset (single-modal)
    #
    (
        "porestar/seismicfoundationmodel-interpolation",
        ["train", "validation"], 
        {"train": (224, 224), "validation": (224, 224)},
        {"train": 3500, "validation": 500}  
    ),
    #
    # Inversion synthetic
    #
    (
        "porestar/seismicfoundationmodel-inversion-synthetic",
        ["train"], 
        {"train": (224, 224)},
        {"train": 2200}  # your example said 0-2199 => 2200 items, "validation" was empty
    ),
    #
    # Inversion seam
    #
    (
        "porestar/seismicfoundationmodel-inversion-seam",
        ["train"], 
        {"train": (224, 224)},
        {"train": 5000}  # your example said 0-4999 => 5000 items
    ),
])
def test_seismicfoundationmodel_datasets(dataset_name, expected_splits, expected_shapes, expected_sizes):
    dataset = load_dataset(dataset_name).with_format(type="numpy")

    # Check splits
    actual_splits = list(dataset.keys())
    for split in expected_splits:
        assert split in actual_splits, (
            f"Expected split '{split}' not found in dataset {dataset_name}. Found splits: {actual_splits}"
        )

    for split in expected_splits:
        if split not in dataset:
            continue

        # If we know the exact number of examples for the split, assert that as well
        if expected_sizes[split] is not None:
            assert len(dataset[split]) == expected_sizes[split], (
                f"Dataset {dataset_name} split '{split}' expected size {expected_sizes[split]}, "
                f"got {len(dataset[split])}"
            )

        # Check shape(s). Single-modal datasets only have "seismic". 
        # Multi-modal typically have "seismic" and "label".
        dsplit = dataset[split]
        if "seismic" in dsplit.features:
            item = dsplit.select([0])["seismic"].squeeze()
            shape = item.shape
            assert shape == expected_shapes[split], (
                f"Dataset {dataset_name}, split '{split}', `seismic` shape mismatch: "
                f"expected {expected_shapes[split]} but got {shape}"
            )
        if "label" in dsplit.features:
            item = dsplit.select([0])["seismic"].squeeze()
            shape = item.shape
            # Usually the same shape as `seismic`, unless you intentionally changed label dtype/shape
            assert shape == expected_shapes[split], (
                f"Dataset {dataset_name}, split '{split}', `label` shape mismatch: "
                f"expected {expected_shapes[split]} but got {shape}"
            )

#
# Example test for the SuperResolution dataset from your original snippet
#
@pytest.mark.parametrize("dataset_name, expected_splits, expected_sizes, lr_shape, hr_shape", [
    (
        "porestar/superresolutiondataset", 
        ["train", "test"], 
        {"train": 1200, "test": 150}, 
        (128, 128),  # LR image shape
        (256, 256),  # HR image shape
    )
])
def test_superresolution_dataset(dataset_name, expected_splits, expected_sizes, lr_shape, hr_shape):
    dataset = load_dataset(dataset_name).with_format("numpy")
    # Check splits exist
    actual_splits = list(dataset.keys())
    for split in expected_splits:
        assert split in actual_splits, (
            f"Expected split '{split}' not found in dataset {dataset_name}. Found splits: {actual_splits}"
        )

    for split in expected_splits:
        dsplit = dataset[split]
        # Check size
        assert len(dsplit) == expected_sizes[split], (
            f"Dataset {dataset_name}, split '{split}' expected size {expected_sizes[split]}, got {len(dsplit)}"
        )
        # Check shapes
        if "lr" in dsplit.features:
            item = dsplit.select([0])["lr"].squeeze()
            actual_lr_shape = item.shape
            assert actual_lr_shape == lr_shape, (
                f"Expected LR shape {lr_shape} in '{split}' split, got {actual_lr_shape}"
            )
        if "hr" in dsplit.features:
            item = dsplit.select([0])["hr"].squeeze()
            actual_hr_shape = item.shape
            assert actual_hr_shape == hr_shape, (
                f"Expected HR shape {hr_shape} in '{split}' split, got {actual_hr_shape}"
            )