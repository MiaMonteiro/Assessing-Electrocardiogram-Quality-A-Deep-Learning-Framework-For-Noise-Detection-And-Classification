import numpy as np
import os
from tools.process_noise import process_noise
from tools.add_noise import add_noise_to_signals_in_folder

# Sampling rate (Hz)
sampling_rate = 360
total_length = 3600

# Access the noise data from the pickle file generated in 1_get_records.py
dir_noise = r'C:\Users\marci\paper_proj_dataset\paper\mit_bih_noise_stress'
bw_raw = np.load(os.path.join(dir_noise, 'bw.npy'))
ma_raw = np.load(os.path.join(dir_noise, 'ma.npy'))
em_raw = np.load(os.path.join(dir_noise, 'em.npy'))
# Process each noise type, apply the function to the last 3 elements of the noise_in list

bw_processed = process_noise(bw_raw)
em_processed = process_noise(em_raw)
ma_processed = process_noise(ma_raw)

# Split data into train, validation, and test sets 70/15/15
# the split is done in two steps to the first split the data into train and temp sets, and then the temp set into val and test

noise_length = len(bw_processed)
ma_train = ma_processed[:int(noise_length*0.8)]
ma_val = ma_processed[int(noise_length*0.8) : int(noise_length*0.9)]
ma_test = ma_processed[int(noise_length*0.9):]

em_train = em_processed[:int(noise_length*0.8)]
em_val = em_processed[int(noise_length*0.8) : int(noise_length*0.9)]
em_test = em_processed[int(noise_length*0.9):]

bw_train = bw_processed[:int(noise_length*0.8)]
bw_val = bw_processed[int(noise_length*0.8) : int(noise_length*0.9)]
bw_test = bw_processed[int(noise_length*0.9):]

# Store train, validation, and test sets in dictionaries
noise_data_train = {'MA': ma_train, 'EM': em_train, 'BW': bw_train}
noise_data_val = {'MA': ma_val, 'EM': em_val, 'BW': bw_val}
noise_data_test = {'MA': ma_test, 'EM': em_test, 'BW': bw_test}

# Main output folder for noisy datasets
main_output_folder = r'C:\Users\marci\paper_proj_dataset\ptb_xl_noisy_360'
# Create the main output folder if it doesn't exist
os.makedirs(main_output_folder, exist_ok=True)

# Clean signals directories
test_folder = r'C:\Users\marci\paper_proj_dataset\ptb_xl_clean_360\x_test_clean'
train_folder = r'C:\Users\marci\paper_proj_dataset\ptb_xl_clean_360\x_train_clean'
val_folder = r'C:\Users\marci\paper_proj_dataset\ptb_xl_clean_360\x_val_clean'

# Adding noise to each set
add_noise_to_signals_in_folder('x_test', test_folder, main_output_folder, noise_data_test)
add_noise_to_signals_in_folder('x_train', train_folder, main_output_folder, noise_data_train)
add_noise_to_signals_in_folder('x_val', val_folder, main_output_folder, noise_data_val)
