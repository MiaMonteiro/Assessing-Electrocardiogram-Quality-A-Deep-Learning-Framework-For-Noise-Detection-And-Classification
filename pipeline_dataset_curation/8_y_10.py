import numpy as np
import os
from tqdm import tqdm
from tools.ensure_folders_empty import ensure_folders_empty


def generate_output_10(noise_info, total_length=3600):
    # Calculate number of seconds in the signal
    num_seconds = total_length // 360
    # Initialize an array of zeros with shape (3, num_seconds)
    output_array = np.zeros((3, num_seconds), dtype=int)

    # Iterate over each interval in noise_info
    for noise in noise_info:
        start_idx = noise[0]
        end_idx = noise[1]
        ma = noise[2]
        em = noise[3]
        bw = noise[4]

        # Ensure indices are within bounds
        start_idx = max(0, start_idx)
        end_idx = min(total_length, end_idx)

        # Determine the range in seconds
        start_sec = start_idx // 360
        end_sec = end_idx // 360

        # Apply the noise information for the given interval allowing overlap
        if ma:
            output_array[0, start_sec:end_sec + 1] = 1
        if em:
            output_array[1, start_sec:end_sec + 1] = 1
        if bw:
            output_array[2, start_sec:end_sec + 1] = 1
    # print(output_array.shape)
    # print(output_array)
    return output_array


def process_noise_info_files(noise_info_folder, output_folder, signal_length=3600):
    # List all files in the noise info folder
    all_files = [f for f in os.listdir(noise_info_folder) if f.endswith('.npy')]

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for file_name in tqdm(all_files, desc=f"Processing files in {noise_info_folder}"):
        noise_info_path = os.path.join(noise_info_folder, file_name)

        # Load the noise info
        noise_info = np.load(noise_info_path, allow_pickle=True)

        # Generate the binary array
        output_array = generate_output_10(noise_info, signal_length)

        # Modify the output filename by removing "_noise_info" suffix
        output_file_name = file_name.replace('_noise_info', '')

        # Save the binary array
        output_path = os.path.join(output_folder, output_file_name)
        np.save(output_path, output_array)


# Directories
x_test_noise_info = r'C:\Users\marci\paper_proj_dataset\ptb_xl_noisy\noise_info\x_test_noise_info'
x_train_noise_info = r'C:\Users\marci\paper_proj_dataset\ptb_xl_noisy\noise_info\x_train_noise_info'
x_val_noise_info = r'C:\Users\marci\paper_proj_dataset\ptb_xl_noisy\noise_info\x_val_noise_info'

# Output directories
y_test = r'C:\Users\marci\paper_proj_dataset\ptb_xl_y_classification\y_10\y_test'
y_train = r'C:\Users\marci\paper_proj_dataset\ptb_xl_y_classification\y_10\y_train'
y_val = r'C:\Users\marci\paper_proj_dataset\ptb_xl_y_classification\y_10\y_val'
#make the directories
os.makedirs(y_test, exist_ok=True)
os.makedirs(y_train, exist_ok=True)
os.makedirs(y_val, exist_ok=True)

process_noise_info_files(x_test_noise_info, y_test)
process_noise_info_files(x_train_noise_info, y_train)
process_noise_info_files(x_val_noise_info, y_val)

