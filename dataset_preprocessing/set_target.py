import os
import shutil
import numpy as np

target = "acc"     # displacements, velocity_differences, accelerations, acc_new, acc
dataset_ver = "test_set"
num_of_iterations = 1
frames_per_iteration = 500
fps = 10
delta_t = 1 / fps

# --- Custom Digit Setting ---
num_digits = 4     # Change this to 3 for 001_001, or 4 for 0001_0001, etc.

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
elif target == "acc_new":
    target_path = os.path.join(target_path, "acc_new")
    print("targeting kinematic acceleration (acc_new)...")
elif target == "acc":
    target_path = os.path.join(target_path, "acc")
    print("targeting kinematic acceleration (acc)...")
else:
    raise ValueError("Invalid target type specified.")


if os.path.exists(target_path):
    shutil.rmtree(target_path)
os.makedirs(target_path, exist_ok=False)

for i in range (1, num_of_iterations+1):
    for j in range (0, frames_per_iteration):
        print(f"\r Processing iteration {i}, frame {j}...", end="", flush=True)
        if target == "displacements":
            first = np.load(dataset_path + f'/flag_{i:0{num_digits}d}_{j:0{num_digits}d}.npy')
            second = np.load(dataset_path + f'/flag_{i:0{num_digits}d}_{j+1:0{num_digits}d}.npy')
            displacement = second[:,:3] - first[:,:3]
            np.save(target_path + f'/target_{i:0{num_digits}d}_{j:0{num_digits}d}.npy', displacement)
            
        elif target == "velocity_differences":
            first = np.load(dataset_path + f'/flag_{i:0{num_digits}d}_{j:0{num_digits}d}.npy')
            second = np.load(dataset_path + f'/flag_{i:0{num_digits}d}_{j+1:0{num_digits}d}.npy')
            velocity_difference = second[:,3:] - first[:,3:]
            np.save(target_path + f'/target_{i:0{num_digits}d}_{j:0{num_digits}d}.npy', velocity_difference)
            
        elif target == "accelerations":
            first = np.load(dataset_path + f'/flag_{i:0{num_digits}d}_{j:0{num_digits}d}.npy')
            second = np.load(dataset_path + f'/flag_{i:0{num_digits}d}_{j+1:0{num_digits}d}.npy')
            acceleration = (second[:,3:] - first[:,3:]) / delta_t
            np.save(target_path + f'/target_{i:0{num_digits}d}_{j:0{num_digits}d}.npy', acceleration)
            
        elif target == "acc_new":
            if j == 0:
                curr = np.load(dataset_path + f'/flag_{i:0{num_digits}d}_{j:0{num_digits}d}.npy')
                acc_new = np.zeros_like(curr[:, :3])
            else:
                prev = np.load(dataset_path + f'/flag_{i:0{num_digits}d}_{j-1:0{num_digits}d}.npy')
                curr = np.load(dataset_path + f'/flag_{i:0{num_digits}d}_{j:0{num_digits}d}.npy')
                next_f = np.load(dataset_path + f'/flag_{i:0{num_digits}d}_{j+1:0{num_digits}d}.npy')
                
                acc_new = (next_f[:,:3] - 2 * curr[:,:3] + prev[:,:3]) / (0.5 * (delta_t ** 2))
                
            np.save(target_path + f'/target_{i:0{num_digits}d}_{j:0{num_digits}d}.npy', acc_new)
        
        elif target == "acc":
            if j == 0:
                curr = np.load(dataset_path + f'/flag_{i:0{num_digits}d}_{j:0{num_digits}d}.npy')
                acc = np.zeros_like(curr[:, :3])
            else:
                prev = np.load(dataset_path + f'/flag_{i:0{num_digits}d}_{j-1:0{num_digits}d}.npy')
                curr = np.load(dataset_path + f'/flag_{i:0{num_digits}d}_{j:0{num_digits}d}.npy')
                next_f = np.load(dataset_path + f'/flag_{i:0{num_digits}d}_{j+1:0{num_digits}d}.npy')
                
                # MGN Simplified Target (assuming dt=1)
                acc = next_f[:,:3] - 2 * curr[:,:3] + prev[:,:3]
                
            np.save(target_path + f'/target_{i:0{num_digits}d}_{j:0{num_digits}d}.npy', acc)
            
        else:
            raise ValueError("Invalid target type specified.")

print(f"\ntargets saved to {target_path}")