import os
import shutil
import numpy as np

target = "accelerations"     # displacements, velocity_differences, accelerations
dataset_ver = 6
num_of_iterations = 100
frames_per_iteration = 300
fps = 10
delta_t = 1 / fps

dataset_path = f"../../datasets/{str(dataset_ver)}/flags"
target_path = f"../../datasets/{str(dataset_ver)}/targets"

if target == "displacements":
    target_path = os.path.join(target_path, "displacements")
    print("targeting displacement...")
elif target == "velocity_differences":
    target_path = os.path.join(target_path, "velocity_differences")
    print("targeting velocity difference...")
elif target == "accelerations":
    target_path = os.path.join(target_path, "accelerations")
    print("targeting acceleration...")

if os.path.exists(target_path):
    shutil.rmtree(target_path)
os.makedirs(target_path, exist_ok=False)

for i in range (1, num_of_iterations+1):
    for j in range (0, frames_per_iteration):
        print(f"\r Processing iteration {i}, frame {j}...", end="", flush=True)
        if target == "displacements":
            first = np.load(dataset_path + f'/flag_{i:03d}_{j:03d}.npy')
            second = np.load(dataset_path + f'/flag_{i:03d}_{j+1:03d}.npy')
            displacement = second[:,:3] - first[:,:3]
            np.save(target_path + f'/target_{i:03d}_{j:03d}.npy', displacement)
        elif target == "velocity_differences":
            first = np.load(dataset_path + f'/flag_{i:03d}_{j:03d}.npy')
            second = np.load(dataset_path + f'/flag_{i:03d}_{j+1:03d}.npy')
            velocity_difference = second[:,3:] - first[:,3:]
            np.save(target_path + f'/target_{i:03d}_{j:03d}.npy', velocity_difference)
        elif target == "accelerations":
            first = np.load(dataset_path + f'/flag_{i:03d}_{j:03d}.npy')
            second = np.load(dataset_path + f'/flag_{i:03d}_{j+1:03d}.npy')
            acceleration = (second[:,3:] - first[:,3:]) / delta_t
            np.save(target_path + f'/target_{i:03d}_{j:03d}.npy', acceleration)
        else:
            raise ValueError("Invalid target type specified.")

print(f"targets saved to {target_path}")