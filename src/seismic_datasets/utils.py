import os 
import numpy as np

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
