import numpy as np
import os
import config as cfg  # Import settings from config.py

def generate_and_save_topology():
    edges = []
    H, W = cfg.PHYSICS_CONFIG['grid_h'], cfg.PHYSICS_CONFIG['grid_w']
    
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
    save_path = os.path.join(cfg.topology_output_folder, "topology_edge_index.npy")
    np.save(save_path, edge_index)
    print(f"✅ Topology saved to {save_path}")

def get_cube_wind(pos, wind_8_vectors):
    x, y, z = pos
    idx_x = 1 if x > cfg.MID_X else 0
    idx_y = 1 if y > cfg.MID_Y else 0
    idx_z = 1 if z > cfg.MID_Z else 0
    cube_index = (idx_x * 4) + (idx_y * 2) + idx_z
    return wind_8_vectors[cube_index]

# def compute_aero_force(velocity_cloth, velocity_wind, normal_vector):
#     v_rel = velocity_wind - velocity_cloth
#     v_mag = np.linalg.norm(v_rel)
#     if v_mag < 1e-6: return np.zeros(3)
#     v_dir = v_rel / v_mag
#     cos_theta = np.dot(v_dir, normal_vector)
    
#     area = cfg.SPACING_W * cfg.SPACING_H
#     force = 0.5 * 1.225 * (v_mag**2) * 1.5 * np.abs(cos_theta) * area * v_dir
#     return force

def compute_drag_lift_vectorized(pos_all, vel_all, triangles, wind_vectors):
    """
    Simulates ARCSim-style aerodynamics on triangle faces.
    
    pos_all: (N, 3) array of all node positions
    vel_all: (N, 3) array of all node velocities
    triangles: (M, 3) array of node indices forming triangles
    wind_vectors: (M, 3) wind vector for each triangle center
    """
    
    # 1. Get positions and velocities of triangle vertices
    # Shape: (M, 3) for p0, p1, p2
    p0 = pos_all[triangles[:, 0]]
    p1 = pos_all[triangles[:, 1]]
    p2 = pos_all[triangles[:, 2]]
    
    v0 = vel_all[triangles[:, 0]]
    v1 = vel_all[triangles[:, 1]]
    v2 = vel_all[triangles[:, 2]]
    
    # 2. Compute Triangle Properties
    # Surface Velocity (Average of 3 corners)
    surface_vel = (v0 + v1 + v2) / 3.0
    
    # Surface Normal (Cross Product of edges)
    # Edge vectors
    u = p1 - p0
    v = p2 - p0
    normals = np.cross(u, v)
    
    # Area of triangles (0.5 * magnitude of cross product)
    areas = 0.5 * np.linalg.norm(normals, axis=1, keepdims=True)
    
    # Normalize normals
    # Add small epsilon to avoid division by zero
    norms_mag = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    normals = normals / norms_mag
    
    # 3. Relative Wind Velocity
    # v_rel = v_wind - v_surface
    v_rel = wind_vectors - surface_vel
    v_rel_mag = np.linalg.norm(v_rel, axis=1, keepdims=True) + 1e-8
    v_rel_dir = v_rel / v_rel_mag
    
    # 4. Angle of Attack (Cos Theta)
    # Dot product of Wind Direction and Surface Normal
    # Shape: (M, 1)
    cos_theta = np.sum(v_rel_dir * normals, axis=1, keepdims=True)
    
    # 5. Compute Lift and Drag Forces
    # Coefficients (Tuned for cloth)
    # Drag (Cd) acts parallel to wind
    # Lift (Cl) acts perpendicular to wind
    
    # Standard aero formula: F = 0.5 * rho * |v|^2 * Area * Coeff
    rho = 1.225
    dynamic_pressure = 0.5 * rho * (v_rel_mag ** 2) * areas
    
    # Drag Component: Resists the wind
    # Effective area seen by wind is Area * sin(theta) roughly, 
    # but standard drag uses projected area.
    # Simple model: Drag is max when perpendicular (cos_theta ~ 1)
    drag_coeff = 1.2
    # Drag acts in direction of relative wind
    f_drag = dynamic_pressure * drag_coeff * v_rel_dir 
    
    # Lift Component: Perpendicular to airflow
    # Lift is max at 45 degrees usually. 
    # Simple model: Lift acts along the normal vector.
    # sin(2*theta) approximation or just proportional to angle.
    lift_coeff = 0.8
    # Lift acts along the surface normal
    # We multiply by sin(theta) * cos(theta) to simulate stall at 0 and 90 degrees
    # Or simpler: cross_flow magnitude
    
    # ARCSim simplified lift: Force along normal proportional to angle
    f_lift = dynamic_pressure * lift_coeff * cos_theta * normals
    
    total_force_per_triangle = f_drag + f_lift
    
    # 6. Distribute Force back to Nodes
    # Each vertex gets 1/3 of the triangle's force
    return total_force_per_triangle / 3.0