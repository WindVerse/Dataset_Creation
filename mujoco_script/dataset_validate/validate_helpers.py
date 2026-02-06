import numpy as np
import os
import v_config as cfg

def generate_and_save_topology():
    """
    Generates topology for a COLUMN-MAJOR grid (MuJoCo flexcomp default).
    
    Index Formula: idx = col * H + row
    (Iterates down column 0, then down column 1, etc.)
    """
    edges = []
    H, W = cfg.PHYSICS_CONFIG['grid_h'], cfg.PHYSICS_CONFIG['grid_w']
    
    # Helper to get index in Column-Major order
    def get_idx(r, c):
        return c * H + r

    # 1. Horizontal Edges (Left <-> Right)
    # Connect (r, c) to (r, c+1)
    for r in range(H):
        for c in range(W - 1):
            curr = get_idx(r, c)
            right = get_idx(r, c + 1)
            
            edges.append([curr, right])
            edges.append([right, curr])

    # 2. Vertical Edges (Top <-> Bottom)
    # Connect (r, c) to (r+1, c)
    # Note: In Column-Major, (r+1) is just (curr + 1) because we go down columns first.
    for r in range(H - 1):
        for c in range(W):
            curr = get_idx(r, c)
            below = get_idx(r + 1, c)
            
            edges.append([curr, below])
            edges.append([below, curr])

    # Save
    edge_index = np.array(edges).T
    save_path = os.path.join(cfg.topology_output_folder, "topology_edge_index.npy")
    np.save(save_path, edge_index)
    
    print(f"✅ Topology saved to {save_path}")
    print(f"   Nodes: {H*W}, Edges: {edge_index.shape[1]}")

def get_triangle_indices(H, W):
    """
    Generates triangle indices for a COLUMN-MAJOR grid to match MuJoCo flexcomp.
    Formula: idx = col * H + row
    """
    indices = []
    
    # Helper to ensure consistency
    def get_idx(r, c):
        return c * H + r

    for r in range(H - 1):
        for c in range(W - 1):
            # Calculate indices using Column-Major logic
            # (r, c)     (r, c+1)
            #   TL -------- TR
            #   |         / |
            #   |       /   |
            #   |     /     |
            #   BL -------- BR
            # (r+1, c)   (r+1, c+1)

            tl = get_idx(r, c)          # Top-Left
            tr = get_idx(r, c + 1)      # Top-Right
            bl = get_idx(r + 1, c)      # Bottom-Left
            br = get_idx(r + 1, c + 1)  # Bottom-Right
            
            # Triangle 1 (Top-Left, Bottom-Left, Top-Right)
            # Note: The winding order (CCW vs CW) determines the normal direction.
            # Standard CCW:
            indices.append([tl, bl, tr])
            
            # Triangle 2 (Top-Right, Bottom-Left, Bottom-Right)
            indices.append([tr, bl, br])
            
    return np.array(indices, dtype=np.int32)


def save_obj(vertices, faces, filename):
    """
    Saves mesh data to a .obj file compatible with Unity/Blender.
    vertices: (N, 3) numpy array of positions
    faces: (M, 3) numpy array of triangle indices (0-based)
    """
    with open(filename, 'w') as f:
        f.write("# Initial Flag Mesh for Unity\n")
        # 1. Write Vertices (v x y z)
        # Unity uses a different coordinate system (Y-up, Left-handed), 
        # but standard export (Z-up) usually imports fine with -90 rotation x-form.
        for v in vertices:
            # Writing standard MuJoCo coordinates (usually Z-up)
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
        # 2. Write Faces (f v1 v2 v3)
        # Note: OBJ indices are 1-based, Python is 0-based
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            
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
    Simulates aerodynamics using Thin Airfoil Theory.
    
    Upgrades from previous version:
    1. Variable Drag Coefficient (based on projected area).
    2. Variable Lift Coefficient (based on stall angle).
    3. Correct Lift Direction (Perpendicular to airflow, not surface normal).
    """
    
    # 1. Get positions and velocities of triangle vertices
    p0 = pos_all[triangles[:, 0]]
    p1 = pos_all[triangles[:, 1]]
    p2 = pos_all[triangles[:, 2]]
    
    v0 = vel_all[triangles[:, 0]]
    v1 = vel_all[triangles[:, 1]]
    v2 = vel_all[triangles[:, 2]]
    
    # 2. Surface Properties
    # Surface Velocity
    surface_vel = (v0 + v1 + v2) / 3.0
    
    # Surface Normal & Area
    u = p1 - p0
    v = p2 - p0
    cross_product = np.cross(u, v)
    
    # Area = 0.5 * magnitude of cross product
    # Add epsilon to avoid division by zero
    norms_mag = np.linalg.norm(cross_product, axis=1, keepdims=True) + 1e-10
    areas = 0.5 * norms_mag
    normals = cross_product / norms_mag
    
    # 3. Relative Wind Velocity
    # v_rel = v_wind - v_surface
    v_rel = wind_vectors - surface_vel
    v_rel_mag = np.linalg.norm(v_rel, axis=1, keepdims=True) + 1e-10
    v_rel_dir = v_rel / v_rel_mag
    
    # 4. Aerodynamic Geometry (Thin Airfoil Theory)
    # Cosine of angle between Wind and Normal (phi)
    # If cos_phi = 1, wind is hitting FACE-ON.
    # If cos_phi = 0, wind is hitting EDGE-ON (Parallel).
    cos_phi = np.sum(v_rel_dir * normals, axis=1, keepdims=True)
    
    # Clamp for numerical stability [-1, 1]
    cos_phi = np.clip(cos_phi, -1.0, 1.0)
    
    # Sin(phi) derived from Cos(phi)
    sin_phi = np.sqrt(1.0 - cos_phi**2)
    
    # 5. Compute Coefficients (ARCSim Model)
    # ---------------------------------------------------------
    # Drag Coefficient (Cd): 
    # Max when face-on (cos_phi=1), Min when edge-on.
    # We use Projected Area approximation: Cd ~ |cos_phi|
    Cd_base = 1.2
    Cd = Cd_base * np.abs(cos_phi)
    
    # Lift Coefficient (Cl):
    # Max at 45 degrees, Zero at 0 (face-on) and 90 (edge-on).
    # Approx: sin(2 * alpha) where alpha is angle of attack.
    # Since phi is angle with normal, sin(2*alpha) ~ 2 * sin(phi) * cos(phi)
    Cl_base = 0.8
    Cl = Cl_base * sin_phi * cos_phi 
    
    # 6. Force Directions
    # ---------------------------------------------------------
    # Drag Direction: Always parallel to relative wind
    dir_drag = v_rel_dir
    
    # Lift Direction: Perpendicular to Wind AND lying in the plane defined by Normal and Wind.
    # Computed as: (Normal x Wind) x Wind
    # This creates a vector orthogonal to wind, "slicing" the air.
    
    # n_cross_v is the vector sticking out of the 2D plane of flow
    n_cross_v = np.cross(normals, v_rel_dir)
    
    # Crossing it back with v gives the orthogonal lift vector
    dir_lift = np.cross(n_cross_v, v_rel_dir)
    
    # Normalize lift direction (safe division)
    lift_norm = np.linalg.norm(dir_lift, axis=1, keepdims=True) + 1e-10
    dir_lift = dir_lift / lift_norm
    
    # 7. Final Force Calculation
    # F = 0.5 * rho * v^2 * Area * Coeff
    rho = 1.225
    dynamic_pressure = 0.5 * rho * (v_rel_mag ** 2) * areas
    
    f_drag = dynamic_pressure * Cd * dir_drag
    f_lift = dynamic_pressure * Cl * dir_lift
    
    total_force = f_drag + f_lift
    
    # 8. Distribute to Nodes (1/3 per vertex)
    return total_force / 3.0


def compute_drag_lift_vectorized_arcsim(pos_all, vel_all, triangles, wind_vectors):
    """
    Simulates aerodynamics using EXACT ARCSim logic from physics.cpp.
    
    Logic:
    F_total = F_normal + F_tangential
    
    1. F_normal = density * Area * |vn| * vn * Normal
       (Acts like pressure/sailing)
       
    2. F_tangential = drag_coeff * Area * vt
       (Acts like surface friction)
    """
    
    # -----------------------------------------------------------
    # 1. SETUP GEOMETRY (Same as before)
    # -----------------------------------------------------------
    # Get positions and velocities of triangle vertices
    p0 = pos_all[triangles[:, 0]]
    p1 = pos_all[triangles[:, 1]]
    p2 = pos_all[triangles[:, 2]]
    
    v0 = vel_all[triangles[:, 0]]
    v1 = vel_all[triangles[:, 1]]
    v2 = vel_all[triangles[:, 2]]
    
    # Surface Velocity (Average of 3 corners)
    # C++: Vec3 vface = (face->v[0]->node->v + ...)/3.
    surface_vel = (v0 + v1 + v2) / 3.0
    
    # Surface Normal & Area
    u = p1 - p0
    v = p2 - p0
    cross_product = np.cross(u, v)
    
    # Area = 0.5 * magnitude of cross product
    # C++: face->a
    norms_mag = np.linalg.norm(cross_product, axis=1, keepdims=True) + 1e-10
    areas = 0.5 * norms_mag
    normals = cross_product / norms_mag # C++: face->n
    
    # -----------------------------------------------------------
    # 2. ARCSIM EXACT LOGIC
    # -----------------------------------------------------------
    
    # A. Relative Velocity
    # C++: Vec3 vrel = wind.velocity - vface;
    v_rel = wind_vectors - surface_vel
    
    # B. Normal Component (vn)
    # C++: double vn = dot(face->n, vrel);
    # (Scalar projection of velocity onto normal)
    vn = np.sum(v_rel * normals, axis=1, keepdims=True)
    
    # C. Tangential Component (vt)
    # C++: Vec3 vt = vrel - vn*face->n;
    vt = v_rel - (vn * normals)
    
    # D. Force Calculation
    # -------------------------------------------------------
    # Constants from ARCSim (You can tune these)
    # wind.density (Air Density)
    air_density = cfg.PHYSICS_CONFIG['air_density'] 
    
    # wind.drag (Tangential Friction Coefficient)
    # NOTE: In ARCSim C++, this is 0 as default, meaning no tangential drag. We can set it to a small value for realism.
    drag_coeff = cfg.PHYSICS_CONFIG['drag_coeff']
    
    # Force 1: Normal (Pressure)
    # C++: wind.density * face->a * abs(vn) * vn * face->n
    # Note: abs(vn)*vn ensures the force pushes in the correct direction 
    # (front vs back) while maintaining quadratic scaling (v^2).
    f_normal = air_density * areas * np.abs(vn) * vn * normals
    
    # Force 2: Tangential (Friction)
    # C++: wind.drag * face->a * vt
    # Note: This scales LINEARLY with velocity, acting as damping.
    f_tangential = drag_coeff * areas * vt
    
    # Total Force
    total_force_per_triangle = f_normal + f_tangential
    
    # -----------------------------------------------------------
    # 3. DISTRIBUTE TO NODES
    # -----------------------------------------------------------
    # C++: fext[face->v[v]->node->index] += fw/3.
    return total_force_per_triangle / 3.0