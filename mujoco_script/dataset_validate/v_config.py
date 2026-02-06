import os
import numpy as np

# ==========================================
# SIMULATION SETTINGS
# ==========================================
TEST = True

if TEST:
    DATASET_VERSION = "temp"
    WIND_ARRAY = [[3, 3, 3]]
    FPS = 10
    SECONDS = 5
    MAX_FRAMES = FPS * SECONDS
    SUBSTEPS = 100
else:
    DATASET_VERSION = "1"
    WIND_ARRAY = [[0, 0, 0], [3, 3, 3], [5, 0, 5], [10, -2, 3], [-12, 0, 2]]
    FPS = 10
    SECONDS = 5
    MAX_FRAMES = FPS * SECONDS
    SUBSTEPS = 100

# ==========================================
# PHYSICS PROPERTIES
# ==========================================

PHYSICS_CONFIG = {
        
    "air_density": 1.225,
    "drag_coeff": 0.1,
    

    # Simulation Precision (Lower is more stable)
    "timestep": 1 / FPS / SUBSTEPS,

    # Environment
    "viscosity": 0.3,     # Air thickness

    # Cloth Material (Nylon-like)
    "node_mass": 0.005,    # Should be 0.0001 to get 250 GSM
    "solref": f"{2*(1 / FPS / SUBSTEPS)} 1",

    # Springs (Tendons)
    "damping": 1.0,       # Resistance to vibration
    
    "poisson": 0.403,      # Poisson's Ratio
    "thickness": 0.0005,   # Shell Thickness (meters)
    "young": 85242.0,      # Young's Modulus (Elasticity)

    # Flag Geometry
    "grid_h": 41,         # Rows
    "grid_w": 61,         # Cols
    "height_m": 2,      # Total Height (meters)
    "width_m": 3,       # Total Width (meters)
    "start_y": 0.0,       # Y-height of bottom row
    "start_x": 0.0,       # X-position of leftmost column
}

# Calculated Spacing
SPACING_H = PHYSICS_CONFIG["height_m"] / (PHYSICS_CONFIG["grid_h"] - 1)
SPACING_W = PHYSICS_CONFIG["width_m"] / (PHYSICS_CONFIG["grid_w"] - 1)


# Wind Cube Logic Midpoints
MID_X, MID_Y, MID_Z = 0.0, 0.0, 0.0


# ==========================================
# PATHS
# ==========================================

dataset_path = "../../../datasets/validation/"
wind_data_file = os.path.join(dataset_path, "wind/wind_data_5d_array.npz")
output_folder = os.path.join(dataset_path, DATASET_VERSION)



flag_output_folder = os.path.join(output_folder, "flags")
flag_obj_folder = os.path.join(output_folder, "flag_objs")
topology_output_folder = os.path.join(output_folder, "topology")



# Create folders immediately when config is imported
for folder in [flag_output_folder, topology_output_folder, flag_obj_folder]:
    os.makedirs(folder, exist_ok=True)