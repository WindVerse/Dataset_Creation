import mujoco
import numpy as np
import os
import sys

import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import our custom modules
import validate_helpers as help

run = "002"
dataset_path_old = os.path.join("../../../../datasets/validation/grid_search/", run)
frame_rate = 2
max_frames = 10

SUBSTEPS = 100 * (10// frame_rate)  # Adjust substeps based on desired frame rate (10 FPS default)
Winds = [[0, 0, 0], [3, 3, 3], [5, 0, 5], [10, -2, 3], [-12, 0, 2]]

viscosities = [0.1, 0.2, 0.3]  # Air thickness
node_masses = [0.0005, 0.001, 0.005]  # Should be 0.0001 to get 250 GSM
dampings = [0.5, 1.0, 1.5]  # Resistance to vibration
poissons = [0.2, 0.403, 0.5]  # Poisson's Ratio
thicknesses = [0.0001, 0.0005, 0.001]  # Shell Thickness (meters)
youngs = [40000, 85242.0, 100000]  # Young's Modulus (Elasticity)
solrefs = [f"{0.001} 1", f"{0.002} 1"]



# Flag Geometry
grid_h = 41         # Rows
grid_w = 61         # Cols
height_m = 2        # Total Height (meters)
width_m = 3         # Total Width (meters)
start_y = 0.0       # Y-height of bottom row
start_x = 0.0       # X-position of leftmost column


def get_model_xml_explicit(viscosity, node_mass, damping, poisson, thickness, young, solref):
    
    # 1. Spacing Calculation
    spacing_x = width_m/ (grid_w - 1)
    spacing_y = height_m / (grid_h - 1)

    # 2. Position & Orientation Logic
    # ---------------------------------------------------------
    # GOAL: X range [0, 0.6], Z range [-0.2, 0.2]
    #
    # A. CENTER X:
    # Since the grid is width 3 centered at 0 (range -1.5 to 1.5),
    # we shift it right by half the width (1.5) to get range 0.0 to 3.0.
    center_x = width_m / 2.0  # = 1.5
    
    # B. make y range [0, 2].
    # we shift it up by half the height (1.0) to get range 0.0 to 2.0.
    center_y = start_y + (height_m / 2.0) # = 1.0
    
    xml = f"""
    <mujoco model="flag_flex">
        <compiler angle="degree"/>
        <option timestep="0.01" integrator="implicitfast" viscosity="{viscosity}" gravity="0 -9.81 0" solver="CG" tolerance="1e-6"/>
        
        <extension>
            <plugin plugin="mujoco.elasticity.shell"/>
        </extension>

        <worldbody>
            <light pos="0 0 10"/>
            <geom name="floor" type="plane" size="10 10 .1" pos="0 0 -1" rgba=".9 .9 .9 1"/>
            
            <body name="flag_root" pos="{center_x} {center_y} 0" euler="0 0 0">
                
                <flexcomp type="grid" name="cloth"
                          count="{grid_w} {grid_h} 1" 
                          spacing="{spacing_x} {spacing_y} 0.01"
                          mass="{node_mass * grid_w * grid_h}" 
                          radius="0.001" rgba="1 0 0 0.3">
                    
                    <edge equality="true" damping="{damping}"/>
                    <contact condim="3" solref="{solref}" solimp=".95 .99 .0001"/>
                    
                    <plugin plugin="mujoco.elasticity.shell">
                        <config key="poisson" value="{poisson}"/>
                        <config key="thickness" value="{thickness}"/>
                        <config key="young" value="{young}"/> 
                    </plugin>
                </flexcomp>
            </body>
        </worldbody>
        
        <equality>
    """

    # 3. STATIC POLE LOGIC (Weld Constraints)
    # Even after rotation, the topology logic (indices) remains the same.
    # We pin Column 0 (the left edge).
    
    H = grid_h
    
    for r in range(H):
        # TRY THIS: Sequential indexing (0, 1, 2 ... H-1)
        # This assumes the nodes 0..H-1 represent the first vertical column (pole).
        node_idx = r
        
        # If the flag looks like it's pinned horizontally along the top instead,
        # then the grid is Row-Major and we need the old logic:
        # node_idx = r * W

        # Explicitly name the pin for easier debugging in the visualizer
        # xml += f'        <weld name="pin_{r}" body1="cloth_{node_idx}" />\n'
        # pin only 1,21,41,... nodes for validation flag
        if r % 10 == 0:
            xml += f'        <weld name="pin_{r}" body1="cloth_{node_idx}" solref="0.001 1" solimp="0.99 0.999 0.001"/>\n'

    xml += """
        </equality>
    </mujoco>
    """
    
    return xml




if __name__ == "__main__":
    
    num_of_possible_combinations = len(viscosities) * len(node_masses) * len(dampings) * len(poissons) * len(thicknesses) * len(youngs) * len(solrefs)
    
    wind = Winds[int(run) - 1]
    print(f"Using wind: {wind}")
    
    H, W = grid_h, grid_w
        
    triangle_indices = help.get_triangle_indices(H, W)
    num_triangles = len(triangle_indices)
    
    help.generate_and_save_topology()

    print("=== STARTING GRID SEARCH ===")
    
    i = 0

    for viscosity in viscosities:
        for node_mass in node_masses:
            for damping in dampings:
                for poisson in poissons:
                    for thickness in thicknesses:
                        for young in youngs:
                            for solref in solrefs:
                                print(f"Running with viscosity={viscosity}, node_mass={node_mass}, damping={damping}, poisson={poisson}, thickness={thickness}, young={young}, solref={solref}")

                                i += 1
                                dataset_path = os.path.join(dataset_path_old, str(i))
                                
                                flag_output_folder = os.path.join(dataset_path, "flags")
                                flag_obj_folder = os.path.join(dataset_path, "flag_objs")
                                
                                if not os.path.exists(flag_output_folder):
                                    os.makedirs(flag_output_folder)
                                if not os.path.exists(flag_obj_folder):
                                    os.makedirs(flag_obj_folder)
                                
                                # save each parameter combination for reference line by line in a text file
                                with open(os.path.join(dataset_path, "parameters.txt"), "a") as f:
                                    f.write(f"viscosity={viscosity}\n")
                                    f.write(f"node_mass={node_mass}\n")
                                    f.write(f"damping={damping}\n")
                                    f.write(f"poisson={poisson}\n")
                                    f.write(f"thickness={thickness}\n")
                                    f.write(f"young={young}\n")
                                    f.write(f"solref={solref}\n")
                                    f.write("\n")
                                    f.close()
                                
                                print("saved parameters to text file")
                                
                                xml_string = get_model_xml_explicit(viscosity, node_mass, damping, poisson, thickness, young, solref)
                                model = mujoco.MjModel.from_xml_string(xml_string)
                                data_sim = mujoco.MjData(model)
                                
                                cloth_ids = []
                                for c in range(W):      # Cols
                                    for r in range(H):  # Rows
                                        seq_idx = c * H + r
                                        name = f"cloth_{seq_idx}" 
                                        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                                        if body_id != -1:
                                            cloth_ids.append(body_id)
                                        else:
                                            print(f"⚠️ Warning: Could not find body {name}")
                                            
        
                                mujoco.mj_resetData(model, data_sim)
                                mujoco.mj_forward(model, data_sim)
                                
                                # Frame 000 (Init)
                                pos_list = [data_sim.xpos[i].copy() for i in cloth_ids]
                                vel_list = [data_sim.cvel[i][:3].copy() for i in cloth_ids]
                                combined = np.hstack((np.array(pos_list), np.array(vel_list)))
                                
                                np.save(os.path.join(flag_output_folder, f"flag_{run}_000.npy"), combined)
                                help.save_obj(np.array(pos_list), triangle_indices, os.path.join(flag_obj_folder, f"flag_{run}_000.obj"))
                                print(f"Frame 000 (Init) Saved")

                                for t in range(max_frames):        
                                    for _ in range(SUBSTEPS):
                                        all_pos = np.array([data_sim.xpos[i] for i in cloth_ids])
                                        all_vel = np.array([data_sim.cvel[i][3:] for i in cloth_ids])
                                        
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
                                            wind_vecs[i] = wind
                                        
                                        # 3. Compute Aero Forces (The heavy lifting)
                                        # returns (M, 3) forces
                                        tri_forces = help.compute_drag_lift_vectorized_arcsim(all_pos, all_vel, triangle_indices, wind_vecs)
                                        
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
                                    
                                    np.save(os.path.join(flag_output_folder, f"flag_{run}_{frame_idx:03d}.npy"), combined)
                                    help.save_obj(np.array(pos_list), triangle_indices, os.path.join(flag_obj_folder, f"flag_{run}_{frame_idx:03d}.obj"))
                                    
                                    # Dynamic progress update
                                    print(f"\r  {str(i)}/{str(num_of_possible_combinations)} | Frame {frame_idx}/{max_frames} Saved", end="", flush=True)

                            print("\n✅ Grid Dataset Generation Complete!")