#this file will read the .npy files and calculate the SNR for each lead and save the SNR as a single value in a ndarray file

import os
import numpy as np
import warnings
from tqdm import tqdm
from tools.calculate_snr import snr, scale_snr


# fui ver o max de SNR que obtemos sem ser o inf e foi 38 ent i capped it at 410
# para alem disso o scaling normalmente nao traduzia bem as transições dos valores por ser uma range muito grande
# por exemplo no modo normal (snr/snr+1)
# snr 0 -> 0
# snr 1 -> 0.5
# snr 9 -> 0.9
# snr 38 -> 0.97
# in order to have a more clear distinction between the values I used a logarithmic scaling
# snr 0 -> 0
# snr 1 -> 0.18
# snr 9 -> 0.62
# snr 38 -> 0.98

def calculate_global_snr_for_folder(input_clean, input_noisy, output_folder, report_path):
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # List all files in the input folder
    files_clean = os.listdir(input_clean)

    # List to store names of files that are skipped due to negative SNR
    skipped_files = []

    # Calculate the SNR for each file with a progress bar
    for file_name in tqdm(files_clean, desc="Calculating SNR"):
        if file_name.endswith('.npy'):
            # Load the original signal from the .npy file
            clean_signal_path = os.path.join(input_clean, file_name)
            clean_signal = np.load(clean_signal_path)

            # Load the corresponding noisy signal
            noisy_signal_path = os.path.join(input_noisy, file_name)
            noisy_signal = np.load(noisy_signal_path)

            # Calculate the SNR
            snr_value = snr(clean_signal, noisy_signal)

            # Check for invalid SNR values
            if np.isnan(snr_value):
                skipped_files.append(f"File: {file_name}, Reason: SNR is NaN, Value: {snr_value}")
                continue
            elif np.isinf(snr_value):
                skipped_files.append(f"File: {file_name}, Reason: SNR is infinite, Value: {snr_value}")
                continue
            elif snr_value < 0:
                skipped_files.append(f"File: {file_name}, Reason: SNR is negative, Value: {snr_value}")
                continue

            # Scale the SNR value if it's non-negative
            scaled_snr_value = scale_snr(snr_value)

            # Save the scaled SNR value in a file
            output_file = os.path.join(output_folder, file_name)
            np.save(output_file, scaled_snr_value)

    # Save the skipped files report
    if skipped_files:
        with open(report_path, 'w') as report_file:
            report_file.write(f"Skipped {len(skipped_files)} files due to invalid SNR:\n")
            report_file.write("\n".join(skipped_files))
            print(f"Report saved to {report_path}")



x_test_input_clean = r'C:\Users\marci\paper_proj_dataset\ptb_xl_clean\x_test_clean'
x_test_input_noisy = r'C:\Users\marci\paper_proj_dataset\ptb_xl_noisy\x_test_noisy'
y_test_output_folder = r'C:\Users\marci\paper_proj_dataset\snr_global\y_test_snr'
y_test_report_path = r'C:\Users\marci\paper_proj_dataset\snr_global\y_test_snr_report.txt'
calculate_global_snr_for_folder(x_test_input_clean, x_test_input_noisy, y_test_output_folder, y_test_report_path)

x_train_input_clean = r'C:\Users\marci\paper_proj_dataset\ptb_xl_clean\x_train_clean'
x_train_input_noisy = r'C:\Users\marci\paper_proj_dataset\ptb_xl_noisy\x_train_noisy'
y_train_output_folder = r'C:\Users\marci\paper_proj_dataset\snr_global\y_train_snr'
y_train_report_path = r'C:\Users\marci\paper_proj_dataset\snr_global\y_train_snr_report.txt'
calculate_global_snr_for_folder(x_train_input_clean, x_train_input_noisy, y_train_output_folder, y_train_report_path)

x_val_input_clean = r'C:\Users\marci\paper_proj_dataset\ptb_xl_clean\x_val_clean'
x_val_input_noisy = r'C:\Users\marci\paper_proj_dataset\ptb_xl_noisy\x_val_noisy'
y_val_output_folder = r'C:\Users\marci\paper_proj_dataset\snr_global\y_val_snr'
y_val_report_path = r'C:\Users\marci\paper_proj_dataset\snr_global\y_val_snr_report.txt'
calculate_global_snr_for_folder(x_val_input_clean, x_val_input_noisy, y_val_output_folder, y_val_report_path)


