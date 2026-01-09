import os
import numpy as np

# ==========================================
# SIMULATION SETTINGS
# ==========================================
DATASET_VERSION = "temp"
NUM_RUNS = 1
FPS = 10
SECONDS = 3
MAX_FRAMES = FPS * SECONDS
SUBSTEPS = 100

# ==========================================
# PHYSICS PROPERTIES
# ==========================================

PHYSICS_CONFIG = {

    # Simulation Precision (Lower is more stable)
    "timestep": 1 / FPS / SUBSTEPS,

    # Environment
    "gravity": "0 0 -9.81",
    "density": 1.2,       # Air density
    "viscosity": 0.3,     # Air thickness
    "integrator": "implicitfast", # Best for cloth

    # Cloth Material (Nylon-like)
    "node_mass": 0.005,    # Should be 0.0001 to get 250 GSM
    "node_radius": 0.001, # Visual size
    # "friction": "0.1 0.1 0.1",
    "solref": f"{2*(1 / FPS / SUBSTEPS)} 1",

    # Springs (Tendons)
    # "stiffness": 200.0,    # Resistance to stretching
    # "bending_stiffness": 10,  # Resistance to bending
    "damping": 1.0,       # Resistance to vibration
    # "width": 0.002,       # Visual thickness

    # Flag Geometry
    "grid_h": 20,         # Rows
    "grid_w": 30,         # Cols
    "height_m": 0.4,      # Total Height (meters)
    "width_m": 0.6,       # Total Width (meters)
    "start_z": 0.2,       # Z-height of top row

    # World
    "floor_z": -100.0
}

# Calculated Spacing
SPACING_H = PHYSICS_CONFIG["height_m"] / (PHYSICS_CONFIG["grid_h"] - 1)
SPACING_W = PHYSICS_CONFIG["width_m"] / (PHYSICS_CONFIG["grid_w"] - 1)


# Wind Cube Logic Midpoints
MID_X, MID_Y, MID_Z = 0.3, 0.0, 0.0


# ==========================================
# PATHS
# ==========================================

dataset_path = "../../datasets/"
wind_data_file = os.path.join(dataset_path, "wind/wind_data_5d_array.npz")
output_folder = os.path.join(dataset_path, DATASET_VERSION)



flag_output_folder = os.path.join(output_folder, "flags")
wind_output_folder = os.path.join(output_folder, "winds")
topology_output_folder = os.path.join(output_folder, "topology")



# Create folders immediately when config is imported
for folder in [flag_output_folder, wind_output_folder, topology_output_folder]:
    os.makedirs(folder, exist_ok=True)

def get_triangle_indices(H, W):
    indices = []
    for r in range(H - 1):
        for c in range(W - 1):
            # Grid node indices
            # Top-Left, Top-Right, Bot-Left, Bot-Right
            tl = r * W + c
            tr = r * W + (c + 1)
            bl = (r + 1) * W + c
            br = (r + 1) * W + (c + 1)
            
            # Triangle 1 (Top-Left, Bottom-Left, Top-Right)
            indices.append([tl, bl, tr])
            # Triangle 2 (Top-Right, Bottom-Left, Bottom-Right)
            indices.append([tr, bl, br])
    return np.array(indices, dtype=np.int32)