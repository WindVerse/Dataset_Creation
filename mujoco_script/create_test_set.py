import mujoco
import numpy as np
import os
import random
import sys
import config as cfg
import helpers as help
import xml_generator as xml_gen

def generate_smooth_wind_series(num_frames, base_wind=(5.0, 6.0, 0), gust_strength=0.5, smoothness=0.95):
    """
    Generates wind by fluctuating magnitude and subtly wobbling direction.
    """
    wind_series = np.zeros((num_frames, 3))
    
    # 1. Calculate Base Magnitude and Direction
    base_mag = np.linalg.norm(base_wind)
    base_dir = np.array(base_wind) / (base_mag + 1e-8)
    
    current_mag = base_mag
    
    # Track a small angle offset for direction wobbling
    current_angle_offset = 0.0 
    
    for t in range(num_frames):
        # 2. Fluctuate Magnitude (Speed)
        mag_noise = np.random.normal(0, gust_strength)
        mag_drift = (base_mag - current_mag) * (1 - smoothness)
        current_mag = current_mag + mag_drift + mag_noise
        
        # 3. Subtle Wobble (Direction)
        # Randomly change the angle offset slowly (low-pass filter)
        angle_noise = np.random.normal(0, 0.05) # Very small directional jitter
        current_angle_offset = (current_angle_offset * 0.9) + angle_noise
        
        # 4. Construct Vector: Base direction + small rotation
        # Rotate base direction slightly based on the offset
        rot_matrix = np.array([
            [np.cos(current_angle_offset), -np.sin(current_angle_offset), 0],
            [np.sin(current_angle_offset),  np.cos(current_angle_offset), 0],
            [0, 0, 1]
        ])
        
        current_dir = rot_matrix @ base_dir
        wind_series[t] = current_dir * current_mag
        
    return wind_series

if __name__ == "__main__":
    # Settings for Test Dataset
    TEST_FRAMES = 1001
    OUTPUT_DIR = "../../datasets/test_set"
    os.makedirs(os.path.join(OUTPUT_DIR, "flags"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "winds"), exist_ok=True)
    
    # Setup Topology
    H, W = cfg.PHYSICS_CONFIG['grid_h'], cfg.PHYSICS_CONFIG['grid_w']
    edge_index, faces_array = help.generate_and_save_topology()
    
    # Initialize MuJoCo
    xml_string = xml_gen.get_model_xml_explicit()
    model = mujoco.MjModel.from_xml_string(xml_string)
    data_sim = mujoco.MjData(model)
    cloth_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"cloth_{c*H + r}") for c in range(W) for r in range(H)]

    # Generate 1 long natural wind sequence (100 seconds)
    natural_wind = generate_smooth_wind_series(TEST_FRAMES, base_wind=(1, 0.4, 0))

    print(f"Starting long-duration simulation ({TEST_FRAMES} frames)...")
    
    mujoco.mj_resetData(model, data_sim)
    
    for t in range(TEST_FRAMES):
        # 1. Get smooth wind for this frame
        current_wind = natural_wind[t]
        # Expand into the 8-cube format model expects
        current_8_winds = np.tile(current_wind, (8, 1))

        for _ in range(cfg.SUBSTEPS):
            all_pos = np.array([data_sim.xpos[i] for i in cloth_ids])
            all_vel = np.array([data_sim.cvel[i][3:] for i in cloth_ids])
            
            # Apply forces
            tri_centers = (all_pos[faces_array[:,0]] + all_pos[faces_array[:,1]] + all_pos[faces_array[:,2]]) / 3.0
            
            # Simplified vectorized wind application
            wind_vecs = np.tile(current_wind, (len(faces_array), 1))
            
            tri_forces = help.compute_drag_lift_vectorized_arcsim(all_pos, all_vel, faces_array, wind_vecs)
            
            node_forces = np.zeros_like(all_pos)
            np.add.at(node_forces, faces_array[:, 0], tri_forces)
            np.add.at(node_forces, faces_array[:, 1], tri_forces)
            np.add.at(node_forces, faces_array[:, 2], tri_forces)
            
            # Apply to MuJoCo
            for i, body_id in enumerate(cloth_ids):
                data_sim.xfrc_applied[body_id][:3] = node_forces[i]
            
            mujoco.mj_step(model, data_sim)
            
        # Save state
        combined = np.hstack((np.array([data_sim.xpos[i].copy() for i in cloth_ids]), 
                              np.array([data_sim.cvel[i][3:].copy() for i in cloth_ids])))
        
        np.save(os.path.join(OUTPUT_DIR, "flags", f"flag_0001_{t:04d}.npy"), combined)
        np.save(os.path.join(OUTPUT_DIR, "winds", f"wind_0001_{t:04d}.npy"), current_8_winds)
        
        print(f"\r Frame {t}/{TEST_FRAMES} saved", end="", flush=True)

    print("\nTest dataset generation complete.")