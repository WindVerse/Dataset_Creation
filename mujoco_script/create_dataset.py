import mujoco
import numpy as np
import os
import random
import sys

# ==========================================
# 1. CONFIGURATION & PHYSICS TUNING
# ==========================================
dataset_version = "6"
NUM_RUNS = 1
FPS = 10
SECONDS = 30
MAX_FRAMES = FPS * SECONDS

# Timestep logic: 0.001s * 100 substeps = 0.1s per frame (10 FPS)
SUBSTEPS = 100 

# --- PHYSICS PROPERTIES ---
PHYSICS_CONFIG = {
    # Simulation Precision (Lower is more stable)
    "timestep": 0.001,
    
    # Environment
    "gravity": "0 0 -9.81",
    "density": 1.2,       # Air density
    "viscosity": 0.5,     # Air thickness (Stability factor)
    "integrator": "implicitfast", # Best for cloth
    
    # Cloth Material (Nylon-like)
    "node_mass": 0.0025,    # Mass per vertex
    "node_radius": 0.008, # Visual size
    "friction": "0.1 0.1 0.1",
    "solref": "0.02 1",   # Compliance (Prevents explosions)
    
    # Springs (Tendons)
    "stiffness": 80.0,    # Resistance to stretching
    "damping": 2.0,       # Resistance to vibration
    "width": 0.002,       # Visual thickness
    
    # Flag Geometry
    "grid_h": 20,         # Rows
    "grid_w": 30,         # Cols
    "height_m": 0.4,      # Total Height (meters)
    "width_m": 0.6,       # Total Width (meters)
    "start_z": 0.2,       # Z-height of top row
    
    # World
    "floor_z": -2.0
}

# Calculated Spacing
SPACING_H = PHYSICS_CONFIG["height_m"] / (PHYSICS_CONFIG["grid_h"] - 1)
SPACING_W = PHYSICS_CONFIG["width_m"] / (PHYSICS_CONFIG["grid_w"] - 1)

# Wind Cube Logic Midpoints
MID_X, MID_Y, MID_Z = 0.3, 0.0, 0.0

# Paths
dataset_path = "C:/Users/janin/Desktop/WindVerse/datasets/"
wind_data_file = os.path.join(dataset_path, "wind/wind_data_5d_array.npz")
output_folder = os.path.join(dataset_path, dataset_version)

flag_output_folder = os.path.join(output_folder, "flags")
wind_output_folder = os.path.join(output_folder, "winds")
topology_output_folder = os.path.join(output_folder, "topology")

for folder in [flag_output_folder, wind_output_folder, topology_output_folder]:
    os.makedirs(folder, exist_ok=True)

# ==========================================
# 2. XML GENERATOR
# ==========================================
def get_model_xml_explicit():
    p = PHYSICS_CONFIG
    
    xml_header = f"""
    <mujoco model="flag_manual">
        <option timestep="{p['timestep']}" gravity="{p['gravity']}" density="{p['density']}" viscosity="{p['viscosity']}" integrator="{p['integrator']}"/>
        <visual><map force="0.1" zfar="30"/></visual>
        
        <default>
            <geom type="sphere" size="{p['node_radius']}" mass="{p['node_mass']}" rgba=".8 .2 .2 1" friction="{p['friction']}" solref="{p['solref']}"/>
            <site type="sphere" size="0.002" rgba="1 1 1 0"/>
        </default>

        <worldbody>
            <light pos="0 0 10"/>
            <geom name="floor" type="plane" size="10 10 .1" pos="0 0 {p['floor_z']}" rgba=".9 .9 .9 1"/>
    """
    
    bodies_xml = ""
    for r in range(p['grid_h']): 
        for c in range(p['grid_w']):
            name = f"B_{r}_{c}"
            site_name = f"S_{r}_{c}"
            
            x = c * SPACING_W
            y = 0
            z = p['start_z'] - (r * SPACING_H)
            
            # --- STATIC POLE LOGIC ---
            # Column 0: No Joint = Static Body (Fixed to World)
            if c == 0:
                joint_xml = "" 
                color_xml = 'rgba=".5 .5 .5 1"'
            else:
                # Column 1+: Free Joint with high damping
                joint_xml = '<joint type="free" damping="0.1"/>' 
                color_xml = 'rgba=".8 .2 .2 1"'

            bodies_xml += f"""
            <body name="{name}" pos="{x} {y} {z}">
                <geom {color_xml}/> 
                <site name="{site_name}" pos="0 0 0"/> 
                {joint_xml}
            </body>
            """
    
    xml_footer_start = "</worldbody>\n<tendon>"
    
    tendons_xml = ""
    
    def make_tendon(s1_name, s2_name):
        return f"""
        <spatial stiffness="{p['stiffness']}" damping="{p['damping']}" width="{p['width']}">
            <site site="{s1_name}"/> <site site="{s2_name}"/>
        </spatial>"""

    # Horizontal
    for r in range(p['grid_h']):
        for c in range(p['grid_w'] - 1):
            tendons_xml += make_tendon(f"S_{r}_{c}", f"S_{r}_{c+1}")

    # Vertical
    for r in range(p['grid_h'] - 1):
        for c in range(p['grid_w']):
            tendons_xml += make_tendon(f"S_{r}_{c}", f"S_{r+1}_{c}")
    
    # Diagonals
    for r in range(p['grid_h'] - 1):
        for c in range(p['grid_w'] - 1):
            tendons_xml += make_tendon(f"S_{r}_{c}", f"S_{r+1}_{c+1}")
            tendons_xml += make_tendon(f"S_{r}_{c+1}", f"S_{r+1}_{c}")

    xml_end = "</tendon>\n</mujoco>"
    
    return xml_header + bodies_xml + xml_footer_start + tendons_xml + xml_end

# ==========================================
# 3. HELPERS
# ==========================================
def generate_and_save_topology():
    edges = []
    H, W = PHYSICS_CONFIG['grid_h'], PHYSICS_CONFIG['grid_w']
    
    for i in range(H):
        for j in range(W - 1):
            idx = i * W + j
            edges.append([idx, idx + 1])
            edges.append([idx + 1, idx])
    for i in range(H - 1):
        for j in range(W):
            idx = i * W + j
            below = (i + 1) * W + j
            edges.append([idx, below])
            edges.append([below, idx])
    edge_index = np.array(edges).T
    save_path = os.path.join(topology_output_folder, "topology_edge_index.npy")
    np.save(save_path, edge_index)
    print(f"✅ Topology saved to {save_path}")

def get_cube_wind(pos, wind_8_vectors):
    x, y, z = pos
    idx_x = 1 if x > MID_X else 0
    idx_y = 1 if y > MID_Y else 0
    idx_z = 1 if z > MID_Z else 0
    cube_index = (idx_x * 4) + (idx_y * 2) + idx_z
    return wind_8_vectors[cube_index]

def compute_aero_force(velocity_cloth, velocity_wind, normal_vector):
    v_rel = velocity_wind - velocity_cloth
    v_mag = np.linalg.norm(v_rel)
    if v_mag < 1e-6: return np.zeros(3)
    v_dir = v_rel / v_mag
    cos_theta = np.dot(v_dir, normal_vector)
    
    area = SPACING_W * SPACING_H
    force = 0.5 * 1.225 * (v_mag**2) * 1.5 * np.abs(cos_theta) * area * v_dir
    return force

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        NUM_RUNS = int(sys.argv[1])
    
    generate_and_save_topology()

    if not os.path.exists(wind_data_file):
        raise FileNotFoundError(f"Cannot find {wind_data_file}")
    data = np.load(wind_data_file, allow_pickle=True)
    wind_data = np.nan_to_num(data["wind_data"])
    n_components, n_z, n_y, n_x, n_time = wind_data.shape
    print(f"✅ Wind Data Loaded: {wind_data.shape}")

    print("🚀 Initializing MuJoCo...")
    xml_string = get_model_xml_explicit()
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)

    cloth_ids = []
    id_map = {}
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and name.startswith("B_"):
            parts = name.split('_')
            r, c = int(parts[1]), int(parts[2])
            id_map[(r,c)] = i

    H, W = PHYSICS_CONFIG['grid_h'], PHYSICS_CONFIG['grid_w']
    for r in range(H):
        for c in range(W):
            if (r,c) in id_map: cloth_ids.append(id_map[(r,c)])

    print(f"🚩 Flag has {len(cloth_ids)} nodes (Should be {H*W}).")

    print("=== STARTING SIMULATIONS ===")

    for run in range(1, NUM_RUNS + 1):
        print(f"--- Run {run}/{NUM_RUNS} ---")
        
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        
        # Frame 000 (Init)
        pos_list = [data.xpos[i].copy() for i in cloth_ids]
        vel_list = [data.cvel[i][:3].copy() for i in cloth_ids]
        combined = np.hstack((np.array(pos_list), np.array(vel_list)))
        
        np.save(os.path.join(flag_output_folder, f"flag_{run:03d}_000.npy"), combined)
        np.save(os.path.join(wind_output_folder, f"wind_{run:03d}_000.npy"), np.zeros((8, 3)))
        print(f"  Run {run} | Frame 000 (Init) Saved")

        # ==========================================
        # --- WIND AUGMENTATION (PER RUN) ---
        # ==========================================
        
        # 1. Random Scale (Wind Strength)
        # scale_factor = random.uniform(0.5, 2.0)
        scale_factor = 5
        
        # 2. Random Rotation (Wind Direction)
        theta = random.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        
        # Rotation Matrix (Rotate around Z-axis)
        rot_matrix = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])

        found_valid_wind = False
        attempts = 0
        start_x, start_y, start_z = 0, 0, 0
        while not found_valid_wind:
            random.seed(run + attempts)
            start_x = random.randint(0, n_x - 2)
            start_y = random.randint(0, n_y - 2)
            start_z = random.randint(0, n_z - 2)
            wind_chunk = wind_data[:, start_z, start_y, start_x, :50]
            if not np.allclose(wind_chunk, 0): found_valid_wind = True
            else: attempts += 1
            if attempts > 100: break
        
        if not found_valid_wind: continue

        for t in range(MAX_FRAMES):
            t_wind = min(t, n_time - 1)
            
            current_8_winds = []
            
            for dx in [0, 1]:
                for dy in [0, 1]:
                    for dz in [0, 1]:
                        vec = wind_data[:, start_z+dz, start_y+dy, start_x+dx, t_wind]
                        
                        # apply augmentation rotation and scaling
                        vec_rotated = rot_matrix @ vec
                        vec_final = vec_rotated * scale_factor
                        current_8_winds.append(vec_final)
            current_8_winds = np.array(current_8_winds)

            for _ in range(SUBSTEPS):
                for body_id in cloth_ids:
                    pos = data.xpos[body_id]
                    vel = data.cvel[body_id][:3]
                    rot = data.xquat[body_id]
                    
                    normal = np.zeros(3)
                    mujoco.mju_rotVecQuat(normal, np.array([0, 0, 1]), rot)
                    
                    wind_vec = get_cube_wind(pos, current_8_winds)
                    force = compute_aero_force(vel, wind_vec, normal)
                    data.xfrc_applied[body_id][:3] = force
                
                mujoco.mj_step(model, data)
            
            frame_idx = t + 1
            pos_list = [data.xpos[i].copy() for i in cloth_ids]
            vel_list = [data.cvel[i][:3].copy() for i in cloth_ids]
            combined = np.hstack((np.array(pos_list), np.array(vel_list)))
            
            np.save(os.path.join(flag_output_folder, f"flag_{run:03d}_{frame_idx:03d}.npy"), combined)
            np.save(os.path.join(wind_output_folder, f"wind_{run:03d}_{frame_idx:03d}.npy"), current_8_winds)
            
            if frame_idx % 50 == 0: print(f"  Run {run} | Frame {frame_idx} Saved")

    print("✅ Dataset Generation Complete!")