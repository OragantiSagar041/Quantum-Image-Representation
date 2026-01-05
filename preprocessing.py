import h5py
import cv2
import numpy as np
import os

def get_all_file_paths(data_dir):
    """Returns a list of all .mat files in the folder."""
    if not os.path.exists(data_dir):
        print(f"Error: Folder '{data_dir}' not found!")
        return []
    return [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.mat')]

def load_medical_data(file_path, size=16):
    """Extracts image and mask from Figshare MATLAB v7.3 (.mat) files."""
    try:
        with h5py.File(file_path, 'r') as f:
            # Figshare v7.3 structure: cjdata/image and cjdata/tumorMask
            # Note: H5PY reads data transposed, so we use .T
            image = np.array(f['cjdata']['image']).T
            mask = np.array(f['cjdata']['tumorMask']).T
            
            # Convert to float for processing
            image = image.astype(float)
            mask = mask.astype(float)
            
            # Resize for Quantum Simulation (2^n)
            img_resized = cv2.resize(image, (size, size))
            mask_resized = cv2.resize(mask, (size, size))
            
            # Normalize 0-1
            img_norm = (img_resized - np.min(img_resized)) / (np.max(img_resized) - np.min(img_resized) + 1e-6)
            
            return img_norm, mask_resized
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None