import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
import random


# Function to process noise
def process_noise(noise):
    concatenated = np.concatenate((np.array(noise[:, 0]), np.array(noise[:, 1])))
    # zscored = zscore(concatenated)  # Z-score normalization
    return concatenated


# Folder path containing .npy files
folder_path = r"C:\Users\marci\paper_proj_dataset\paper\mit_bih_noise_stress"

# Get all .npy file paths
file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".npy")]

# Sampling frequency and time duration
fs = 360  # Hz
segment_length = fs * 10  # 10 seconds in samples

# Initialize subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# Process and plot random 10-second segments for the first three files
# for i, file_path in enumerate(file_paths[:3]):  # Limit to the first three files
#     # Load and process noise
#     noise_data = np.load(file_path)
#     processed_noise = process_noise(noise_data)
#
#     # Determine random start index for 10-second segment
#     max_start_index = len(processed_noise) - segment_length
#     start_index = random.randint(0, max_start_index)
#     end_index = start_index + segment_length
#
#     # Extract 10-second segment
#     segment = processed_noise[start_index:end_index]
#
#     # Create time axis for the segment
#     time_axis = np.linspace(0, 10, segment_length)  # Time in seconds
#
#     # Extract file name without extension
#     file_name = os.path.basename(file_path).replace(".npy", "")
#
#     # Plot the segment
#     axes[i].plot(time_axis, segment, color='black')
#     axes[i].set_title(f"Noise {file_name} (10 seconds)")
#     axes[i].set_ylabel("Z-Score")
#     axes[i].grid(True)
#
# # save as svg
# plt.savefig('noise_signals.svg')
#
# # Add a shared x-axis label
# plt.xlabel("Time (seconds)")
# plt.tight_layout()
# plt.show()

segment_length = fs * 10  # 10 seconds in samples

# Initialize subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# Process and plot the first 10-second segments for the first three files
for i, file_path in enumerate(file_paths[:3]):  # Limit to the first three files
    # Load and process noise
    noise_data = np.load(file_path)
    processed_noise = process_noise(noise_data)

    # Extract the first 10 seconds (segment_length samples)
    segment = processed_noise[:segment_length]

    # Create time axis for the segment
    time_axis = np.linspace(0, 20, segment_length)  # Time in seconds

    # Extract file name without extension
    file_name = os.path.basename(file_path).replace(".npy", "")

    # Plot the segment
    axes[i].plot(time_axis, segment, linewidth=0.5)
    axes[i].set_title(f"Noise {file_name} (First 10 seconds)")
    axes[i].set_ylabel("Z-Score")
    axes[i].grid(True)

# Save the plot as SVG
plt.savefig('noise_signals.svg')

# Add a shared x-axis label
plt.xlabel("Time (seconds)")
plt.tight_layout()
plt.show()