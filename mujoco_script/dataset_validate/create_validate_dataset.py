import json

import mujoco
import numpy as np
import os
import sys
import time

# Import our custom modules
import v_config as cfg
import validate_helpers as help
import generate_validate_xml as xml_gen

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # Save physics config
    with open(os.path.join(cfg.output_folder,"parameters.json"), "w") as file:
        json.dump(cfg.PHYSICS_CONFIG, file, indent=20)
        file.close()
    
    wind_array = cfg.WIND_ARRAY
    
    # save winds
    with open(os.path.join(cfg.output_folder,"winds.txt"), "w") as file:
        for wind in wind_array:
            file.write(str(wind)+"\n")
        file.close()
        
    # Use config length of wind aray to get num of runs
    num_runs = len(wind_array)
    
    H, W = cfg.PHYSICS_CONFIG['grid_h'], cfg.PHYSICS_CONFIG['grid_w']
    
    edge_index, faces_array = help.generate_and_save_topology()
    num_triangles = len(faces_array)
    

    print("Initializing MuJoCo...")
    xml_string = xml_gen.get_model_xml_explicit()
    model = mujoco.MjModel.from_xml_string(xml_string)
    data_sim = mujoco.MjData(model)

    cloth_ids = []
    
    # Loop Columns (x) first, then Rows (y)
    # This ensures cloth_ids[0] corresponds to index 0 in topology
    for c in range(W):      # Cols
        for r in range(H):  # Rows
            
            # Name format depends on generator. 
            # If using flexcomp grid, names are usually "cloth_0", "cloth_1"...
            # If manual bodies, "B_{r}_{c}"...
            
            # Since you are using flexcomp, MuJoCo auto-names them sequentially.
            # We just need to find the ID of "cloth_{i}" where i is the sequential index.
            
            # Calculate the sequential index 
            seq_idx = c * H + r
            name = f"cloth_{seq_idx}" 
            
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            
            if body_id != -1:
                cloth_ids.append(body_id)
            else:
                print(f"Warning: Could not find body {name}")

    print(f"Flag has {len(cloth_ids)} nodes (Should be {H*W}).")

    print("=== STARTING SIMULATIONS ===")

    for run in range(1, num_runs + 1):
        print(f"\n--- Run {run}/{num_runs} ---")
        
        mujoco.mj_resetData(model, data_sim)
        mujoco.mj_forward(model, data_sim)
        
        # Frame 000 (Init)
        pos_list = [data_sim.xpos[i].copy() for i in cloth_ids]
        
        np.save(os.path.join(cfg.flag_output_folder, f"flag_{run:03d}_000.npy"), pos_list)
        help.save_obj(np.array(pos_list), faces_array, os.path.join(cfg.flag_obj_folder, f"flag_{run:03d}_000.obj"))
        print(f"  Run {run} | Frame 000 (Init) Saved")
        
        if run == 1:
            wind_path = os.path.join(cfg.topology_output_folder, "wind.npy")
            np.save(wind_path, wind_array)
            print(f"  Initial Topology OBJ and Wind Array Saved")

        current_wind = wind_array[run - 1]  # Get the wind vector for this run
        print(f"  Run {run} | Wind Vector: {current_wind}")

        Time = 0.0

        for t in range(cfg.MAX_FRAMES):    
            
            t1 = time.time()

            for _ in range(cfg.SUBSTEPS):

                all_pos = np.array([data_sim.xpos[i] for i in cloth_ids])
                all_vel = np.array([data_sim.cvel[i][3:] for i in cloth_ids]) # Linear velocity
                
                # 2. Get Wind for Triangles
                # We need the position of the CENTER of each triangle
                p0 = all_pos[faces_array[:, 0]]
                p1 = all_pos[faces_array[:, 1]]
                p2 = all_pos[faces_array[:, 2]]
                centers = (p0 + p1 + p2) / 3.0
                
                # Vectorized wind lookup (You'll need to update get_cube_wind to handle arrays or loop briefly)
                # Fast loop for wind lookup is okay
                wind_vecs = np.zeros((num_triangles, 3))
                for i in range(num_triangles):
                     wind_vecs[i] = current_wind
                
                # 3. Compute Aero Forces (The heavy lifting)
                # returns (M, 3) forces
                tri_forces = help.compute_drag_lift_vectorized_arcsim(all_pos, all_vel, faces_array, wind_vecs)
                
                # 4. Accumulate Forces on Nodes
                # We need to sum up forces because one node shares multiple triangles
                node_forces = np.zeros_like(all_pos)
                
                # Numpy magic: Add at specific indices (handles duplicates correctly)
                # Add force to Vertex 0 of every triangle
                np.add.at(node_forces, faces_array[:, 0], tri_forces)
                # Add force to Vertex 1
                np.add.at(node_forces, faces_array[:, 1], tri_forces)
                # Add force to Vertex 2
                np.add.at(node_forces, faces_array[:, 2], tri_forces)
                
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
                
                # Dynamic progress update for substeps
                print(f"\r  Run {run} | Frame {t+1}/{cfg.MAX_FRAMES} | Substep {_+1}/{cfg.SUBSTEPS}", end="", flush=True)
                sys.stdout.flush()
                
            t2 = time.time()
            Time += (t2 - t1)
            
            frame_idx = t + 1
            pos_list = [data_sim.xpos[i].copy() for i in cloth_ids]
            
            np.save(os.path.join(cfg.flag_output_folder, f"flag_{run:03d}_{frame_idx:03d}.npy"), pos_list)
            help.save_obj(np.array(pos_list), faces_array, os.path.join(cfg.flag_obj_folder, f"flag_{run:03d}_{frame_idx:03d}.obj"))
            
            # # Dynamic progress update
            # print(f"\r  Run {run} | Frame {frame_idx}/{cfg.MAX_FRAMES} Saved", end="", flush=True)
            
        print(f"\n  Run {run} Completed in {Time:.2f} seconds. Average per frame: {Time/cfg.MAX_FRAMES:.2f} seconds.")

    print("\n✅ Dataset Generation Complete!")