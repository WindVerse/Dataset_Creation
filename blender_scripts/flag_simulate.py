import bpy
import numpy as np
import random
from mathutils import Vector
import os

# --- SETTINGS ---
wind_data_file = "D:/FYP/Codes/wind_data_5d_array.npz"
output_folder = "D:/FYP/BlenderOutputs_Sim"
os.makedirs(output_folder, exist_ok=True)

fps = 10
seconds = 30
cube_size = 1.0
start_frame = 1
end_frame = start_frame + (fps * seconds) - 1
num_runs = 100

flag_obj = bpy.data.objects["Flag"]

# --- LOAD WIND DATA ---
data = np.load(wind_data_file, allow_pickle=True)
wind_data = data["wind_data"]  # shape = (3, z, y, x, t)
n_components, n_z, n_y, n_x, n_time = wind_data.shape
max_time_steps = min(n_time, seconds * fps)

for run in range(1, num_runs + 1):
    print(f"=== RUN {run} ===")

    # --- RANDOM INITIAL ROTATION ---
    random_angle = random.uniform(0, 2 * np.pi)
    flag_obj.rotation_euler.z = random_angle
    flag_obj.keyframe_insert(data_path="rotation_euler", frame=start_frame)

    multiplication_factor = np.random.choice([1, 2, 3, 4, 5])

    # --- CLEANUP PREVIOUS WIND CUBES ---
    for obj in list(bpy.data.objects):
        if obj.name.startswith("WindCube_"):
            bpy.data.objects.remove(obj, do_unlink=True)

    # --- Run till valid wind data ---
    found_valid_wind = False
    attempts = 0
    while not found_valid_wind:
        random.seed(run + attempts)
        x_idx = random.randint(0, n_x-1)
        y_idx = random.randint(0, n_y-1)
        z_idx = random.randint(0, n_z-1)

        wind_block = wind_data[:, z_idx, y_idx, x_idx, :max_time_steps]

        if not np.isnan(wind_block).any() and not np.allclose(wind_block, 0):
            found_valid_wind = True
        else:
            attempts += 1
            if attempts > 100:  # safety to prevent infinite loop
                print(f"⚠️ Run {run}: could not find valid wind data after 100 attempts, skipping.")
                break

    if not found_valid_wind:
        continue

    # --- CREATE WIND CUBES ---
    wind_fields = []
    positions = []

    for dx in [0, 1]:
        for dy in [0, 1]:
            for dz in [0, 1]:
                pos = Vector(((dx - 0.5) * cube_size,
                              (dy - 0.5) * cube_size,
                              (dz - 0.5) * cube_size))
                positions.append(pos)

                bpy.ops.object.effector_add(type='WIND', location=pos)
                obj = bpy.context.object
                obj.name = f"WindCube_{dx}{dy}{dz}"
                obj.field.type = 'WIND'
                obj.field.strength = 0
                obj.field.use_max_distance = True
                obj.field.distance_max = cube_size * 0.5
                obj.field.falloff_type = 'SPHERE'
                obj.field.falloff_power = 0
                wind_fields.append(obj)

    # --- SIMULATION LOOP ---
    for t in range(max_time_steps):
        frame = start_frame + t
        bpy.context.scene.frame_set(frame)
        wind_vectors = []

        for i, wind_obj in enumerate(wind_fields):
            cube_center = positions[i]
            half = cube_size / 2
            min_bound = cube_center - Vector((half, half, half))
            max_bound = cube_center + Vector((half, half, half))

            vertices_inside = [v for v in flag_obj.data.vertices
                               if (min_bound.x <= v.co.x <= max_bound.x and
                                   min_bound.y <= v.co.y <= max_bound.y and
                                   min_bound.z <= v.co.z <= max_bound.z)]

            if vertices_inside:
                xi = min(max(x_idx + (1 if cube_center.x > 0 else 0), 0), n_x - 1)
                yi = min(max(y_idx + (1 if cube_center.y > 0 else 0), 0), n_y - 1)
                zi = min(max(z_idx + (1 if cube_center.z > 0 else 0), 0), n_z - 1)

                u = wind_data[0, zi, yi, xi, t] * multiplication_factor
                v = wind_data[1, zi, yi, xi, t] * multiplication_factor
                w = wind_data[2, zi, yi, xi, t] * multiplication_factor

                wind_vec = Vector((u, v, w))
                strength = wind_vec.length * 100
                direction = wind_vec.normalized() if strength > 1e-6 else Vector((0,0,1))
            else:
                u, v, w = 0.0, 0.0, 0.0
                strength = 0.0
                direction = Vector((0,0,1))

            wind_vectors.append([u, v, w])
            wind_obj.field.strength = strength
            wind_obj.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()
            wind_obj.keyframe_insert(data_path="field.strength", frame=frame)
            wind_obj.keyframe_insert(data_path="rotation_euler", frame=frame)

        # --- EVALUATE FLAG SIMULATION ---
        depsgraph = bpy.context.evaluated_depsgraph_get()
        flag_eval = flag_obj.evaluated_get(depsgraph)
        mesh = flag_eval.to_mesh()
        verts = np.array([flag_obj.matrix_world @ v.co for v in mesh.vertices])
        flag_eval.to_mesh_clear()

        # --- SAVE FLAG AND WIND ---
        flag_save_path = os.path.join(output_folder, f"flag_{run:03d}_{t+1:03d}.npy")
        np.save(flag_save_path, verts)

        wind_save_path = os.path.join(output_folder, f"wind_{run:03d}_{t+1:03d}.npy")
        np.save(wind_save_path, np.array(wind_vectors))

    bpy.context.scene.frame_start = start_frame
    bpy.context.scene.frame_end = end_frame

print("✅ Simulation and flag export complete!")