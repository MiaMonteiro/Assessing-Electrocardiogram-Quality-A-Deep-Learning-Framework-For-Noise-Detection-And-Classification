import os
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing as pp

# Define the paths for the clean and noisy data directories
clean_data_dir = r'C:\Users\marci\paper_proj_dataset\ptb_xl_clean_360'
noisy_data_dir = r'C:\Users\marci\paper_proj_dataset\ptb_xl_noisy_360'

# Define the output directories for the normalized data
clean_data_dir_normalized = r'C:\Users\marci\paper_proj_dataset\ptb_xl_clean_360_normalized'
noisy_data_dir_normalized = r'C:\Users\marci\paper_proj_dataset\ptb_xl_noisy_360_normalized'

# Create the output directories if they don't exist
os.makedirs(clean_data_dir_normalized, exist_ok=True)
os.makedirs(noisy_data_dir_normalized, exist_ok=True)

# Define the specific subfolders to process
clean_subfolders = ['x_train_clean', 'x_test_clean', 'x_val_clean']
noisy_subfolders = ['x_train_noisy', 'x_test_noisy', 'x_val_noisy']

# Function to process and normalize data in specific subfolders
def process_specific_subfolders(data_dir, output_dir, subfolders):
    for subfolder in subfolders:
        input_subfolder = os.path.join(data_dir, subfolder)
        output_subfolder = os.path.join(output_dir, subfolder)
        os.makedirs(output_subfolder, exist_ok=True)

        # Get all the .npy files in the subfolder
        for file_name in tqdm(os.listdir(input_subfolder), desc=f"Normalizing files in {input_subfolder}"):
            if file_name.endswith('.npy'):
                input_file_path = os.path.join(input_subfolder, file_name)
                output_file_path = os.path.join(output_subfolder, file_name)

                # Load the data
                data = np.load(input_file_path)
                normalized_data = pp.minmax_scale(data)

                # Save the normalized data
                np.save(output_file_path, normalized_data)

# Process only the specific clean and noisy subfolders
# process_specific_subfolders(clean_data_dir, clean_data_dir_normalized, clean_subfolders)
process_specific_subfolders(noisy_data_dir, noisy_data_dir_normalized, noisy_subfolders)


