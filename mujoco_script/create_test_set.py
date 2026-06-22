import mujoco
import numpy as np
import os
import random
import sys
import config as cfg
import helpers as help
import xml_generator as xml_gen

def generate_smooth_wind_series(num_frames, base_wind=(5.0, 6.0, 0), gust_strength=1.5, smoothness=0.9):
    wind_series = np.zeros((num_frames, 3))
    
    # Initialize at the base wind
    current_wind = np.array(base_wind, dtype=float)
    
    for t in range(num_frames):
        # Generate random noise centered at 0
        noise = np.random.normal(0, gust_strength, 3)
        
        # Calculate the drift: how far off-base we are
        # We apply the smoothing to the CHANGE, not just the absolute value
        drift = (np.array(base_wind) - current_wind) * (1 - smoothness)
        
        current_wind = current_wind + drift + noise
        
        wind_series[t] = current_wind
        
    return wind_series

if __name__ == "__main__":
    # Settings for Test Dataset
    TEST_FRAMES = 101
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
    natural_wind = generate_smooth_wind_series(TEST_FRAMES, base_wind=(0.5, 0.2, 0))

    print(f"Starting long-duration simulation ({TEST_FRAMES} frames)...")
    
    mujoco.mj_resetData(model, data_sim)
    
    for t in range(TEST_FRAMES):
        # 1. Get smooth wind for this frame
        current_wind = natural_wind[t]
        # Expand into the 8-cube format your model expects
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