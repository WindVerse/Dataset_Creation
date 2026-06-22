import os
import numpy as np

# ==========================================
# SIMULATION SETTINGS
# ==========================================
TEST = False

if TEST:
    DATASET_VERSION = "temp"
    NUM_RUNS = 1
    FPS = 10
    SECONDS = 1
    MAX_FRAMES = FPS * SECONDS
    SUBSTEPS = 100
else:
    DATASET_VERSION = "8"
    NUM_RUNS = 100
    FPS = 10
    SECONDS = 30
    MAX_FRAMES = FPS * SECONDS
    SUBSTEPS = 100

# ==========================================
# PHYSICS PROPERTIES
# ==========================================

PHYSICS_CONFIG = {
    
    "air_density": 1.2,
    "drag_coeff": 0.1,
    

    # Simulation Precision (Lower is more stable)
    "timestep": 1 / FPS / SUBSTEPS,

    # Environment
    "viscosity": 0.2,     # Air thickness

    # Cloth Material (Nylon-like)
    "node_mass": 0.0002,    # Should be 0.0001 to get 250 GSM
    "solref": f"{2*(1 / FPS / SUBSTEPS)} 1",

    # Springs (Tendons)
    "damping": 1.0,       # Resistance to vibration
    
    # from Paper: Evaluating grasp quality metrics of cloth like deformable objects in simulation
    "poisson": 0.403,      # Poisson's Ratio
    "thickness": 0.001,   # Shell Thickness (meters)
    "young": 85242.0,      # Young's Modulus (Elasticity)

    # Flag Geometry
    "grid_h": 20,         # Rows
    "grid_w": 30,         # Cols
    "height_m": 0.4,      # Total Height (meters)
    "width_m": 0.6,       # Total Width (meters)
    "start_z": 0.2,       # Z-height of top row
}

# Calculated Spacing
SPACING_H = PHYSICS_CONFIG["height_m"] / (PHYSICS_CONFIG["grid_h"] - 1)
SPACING_W = PHYSICS_CONFIG["width_m"] / (PHYSICS_CONFIG["grid_w"] - 1)


# Wind Cube Logic Midpoints
MID_X, MID_Y, MID_Z = 0.0, 0.0, 0.0


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