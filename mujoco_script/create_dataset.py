import mujoco
import numpy as np
import os
import random
import sys

# Import our custom modules
import config as cfg
import physics_utils as phys
import xml_generator as xml_gen

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # Use config NUM_RUNS unless overridden by CLI argument
    num_runs = cfg.NUM_RUNS
    if len(sys.argv) > 1:
        num_runs = int(sys.argv[1])
    
    H, W = cfg.PHYSICS_CONFIG['grid_h'], cfg.PHYSICS_CONFIG['grid_w']
        
    triangle_indices = cfg.get_triangle_indices(H, W)
    num_triangles = len(triangle_indices)
    
    phys.generate_and_save_topology()

    if not os.path.exists(cfg.wind_data_file):
        raise FileNotFoundError(f"Cannot find {cfg.wind_data_file}")
    
    data = np.load(cfg.wind_data_file, allow_pickle=True)
    wind_data = np.nan_to_num(data["wind_data"])
    n_components, n_z, n_y, n_x, n_time = wind_data.shape
    print(f"✅ Wind Data Loaded: \n num of components: {n_components} \nx: {n_x}, y: {n_y}, z: {n_z}, \ntime steps: {n_time}")

    print("🚀 Initializing MuJoCo...")
    xml_string = xml_gen.get_model_xml_explicit()
    model = mujoco.MjModel.from_xml_string(xml_string)
    data_sim = mujoco.MjData(model)

    cloth_ids = []
    id_map = {}
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and name.startswith("B_"):
            parts = name.split('_')
            r, c = int(parts[1]), int(parts[2])
            id_map[(r,c)] = i

    for r in range(H):
        for c in range(W):
            if (r,c) in id_map: cloth_ids.append(id_map[(r,c)])

    print(f"🚩 Flag has {len(cloth_ids)} nodes (Should be {H*W}).")

    print("=== STARTING SIMULATIONS ===")

    for run in range(1, num_runs + 1):
        print(f"--- Run {run}/{num_runs} ---")
        
        mujoco.mj_resetData(model, data_sim)
        mujoco.mj_forward(model, data_sim)
        
        # Frame 000 (Init)
        pos_list = [data_sim.xpos[i].copy() for i in cloth_ids]
        vel_list = [data_sim.cvel[i][:3].copy() for i in cloth_ids]
        combined = np.hstack((np.array(pos_list), np.array(vel_list)))
        
        np.save(os.path.join(cfg.flag_output_folder, f"flag_{run:03d}_000.npy"), combined)
        np.save(os.path.join(cfg.wind_output_folder, f"wind_{run:03d}_000.npy"), np.zeros((8, 3)))
        print(f"  Run {run} | Frame 000 (Init) Saved")

        # ==========================================
        # --- WIND AUGMENTATION (PER RUN) ---
        # ==========================================
        
        # 1. Random Scale (Wind Strength)
        scale_factor = random.uniform(0.2, 2)
        # scale_factor = 3
        
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

        for t in range(cfg.MAX_FRAMES):
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

            for _ in range(cfg.SUBSTEPS):
                # 1. Get All Data from MuJoCo (Vectorized)
                # We need full arrays for all bodies
                # This assumes cloth_ids are contiguous or we map them efficiently
                # Best way: Extract all relevant data into numpy arrays
                
                # Create arrays filled with current state
                # Note: This list comprehension is still Python but faster than calculating physics in Python
                all_pos = np.array([data_sim.xpos[i] for i in cloth_ids])
                all_vel = np.array([data_sim.cvel[i][3:] for i in cloth_ids]) # Linear velocity
                
                # 2. Get Wind for Triangles
                # We need the position of the CENTER of each triangle
                p0 = all_pos[triangle_indices[:, 0]]
                p1 = all_pos[triangle_indices[:, 1]]
                p2 = all_pos[triangle_indices[:, 2]]
                centers = (p0 + p1 + p2) / 3.0
                
                # Vectorized wind lookup (You'll need to update get_cube_wind to handle arrays or loop briefly)
                # Fast loop for wind lookup is okay
                wind_vecs = np.zeros((num_triangles, 3))
                for i in range(num_triangles):
                     wind_vecs[i] = phys.get_cube_wind(centers[i], current_8_winds)
                
                # 3. Compute Aero Forces (The heavy lifting)
                # returns (M, 3) forces
                tri_forces = phys.compute_drag_lift_vectorized(all_pos, all_vel, triangle_indices, wind_vecs)
                
                # 4. Accumulate Forces on Nodes
                # We need to sum up forces because one node shares multiple triangles
                node_forces = np.zeros_like(all_pos)
                
                # Numpy magic: Add at specific indices (handles duplicates correctly)
                # Add force to Vertex 0 of every triangle
                np.add.at(node_forces, triangle_indices[:, 0], tri_forces)
                # Add force to Vertex 1
                np.add.at(node_forces, triangle_indices[:, 1], tri_forces)
                # Add force to Vertex 2
                np.add.at(node_forces, triangle_indices[:, 2], tri_forces)
                
                # 5. Apply Safe Clamp & Send to MuJoCo
                MAX_FORCE = 0.05
                force_mags = np.linalg.norm(node_forces, axis=1, keepdims=True)
                # Create mask where force is too high
                unsafe_mask = force_mags > MAX_FORCE
                # Scale down
                scale_factors = np.ones_like(force_mags)
                scale_factors[unsafe_mask] = MAX_FORCE / force_mags[unsafe_mask]
                node_forces *= scale_factors
                
                # Apply to MuJoCo
                for i, body_id in enumerate(cloth_ids):
                    data_sim.xfrc_applied[body_id][:3] = node_forces[i]
                
                mujoco.mj_step(model, data_sim)
            
            frame_idx = t + 1
            pos_list = [data_sim.xpos[i].copy() for i in cloth_ids]
            vel_list = [data_sim.cvel[i][3:].copy() for i in cloth_ids]
            combined = np.hstack((np.array(pos_list), np.array(vel_list)))
            
            np.save(os.path.join(cfg.flag_output_folder, f"flag_{run:03d}_{frame_idx:03d}.npy"), combined)
            np.save(os.path.join(cfg.wind_output_folder, f"wind_{run:03d}_{frame_idx:03d}.npy"), current_8_winds)
            
            # Dynamic progress update
            print(f"\r  Run {run} | Frame {frame_idx}/{cfg.MAX_FRAMES} Saved", end="", flush=True)

    print("\n✅ Dataset Generation Complete!")