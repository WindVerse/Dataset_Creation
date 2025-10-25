import numpy as np
import os

NPY_DIR = "./original"
TXT_DIR = "./converted"
os.makedirs(TXT_DIR, exist_ok=True)

print(f"Converting .npy files from {NPY_DIR} to {TXT_DIR}...")

file_count = 0
for filename in sorted(os.listdir(NPY_DIR)):
    if filename.endswith(".npy"):
        npy_path = os.path.join(NPY_DIR, filename)
        
        # Load the (8, 3) wind data
        wind_data = np.load(npy_path)
        
        # Flatten to a (24,) array
        flat_data = wind_data.flatten()
        
        # Create new .txt filename
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(TXT_DIR, txt_filename)
        
        # Save as space-separated text
        np.savetxt(txt_path, flat_data, fmt='%.6f')
        file_count += 1

print(f"✅ Converted {file_count} files.")