# from tools.calculate_snr import snr, scale_snr
import os
import numpy as np
from tqdm import tqdm
import warnings


def snr(clean_signal, noisy_signal):
    # Implement the SNR calculation
    power_clean = np.sum(clean_signal ** 2)
    power_noise = np.sum((noisy_signal - clean_signal) ** 2)

    if power_noise == 0:
        warnings.warn("Power noise is zero, skipping file.", RuntimeWarning)
        return None  # Return None if there's a division by zero risk

    snr_value = 10 * np.log10(power_clean / power_noise)
    return snr_value


def scale_snr(snr_value, max_snr=40):
    # Scale the SNR value
    scaled_value = np.log10(snr_value + 1) / np.log10(max_snr + 1)
    scaled_value = min(scaled_value, 1)
    return round(scaled_value, 3)


def calculate_global_snr_for_folder(base_clean, base_noisy, noise_info_folder, output_folder):
    skipped_files = []

    # Loop through the clean and noisy data folders
    for set in ['train', 'test', 'val']:
        clean_folder = os.path.join(base_clean, f'x_{set}_clean')
        noisy_folder = os.path.join(base_noisy, f'x_{set}_noisy')
        noise_info_path = os.path.join(noise_info_folder, f'x_{set}_noise_info')

        set_output_folder = os.path.join(output_folder, f'y_{set}_snr_int')
        os.makedirs(set_output_folder, exist_ok=True)

        clean_files = sorted(os.listdir(clean_folder))
        noisy_files = sorted(os.listdir(noisy_folder))
        noise_info_files = sorted(os.listdir(noise_info_path))

        for clean_file, noisy_file, noise_info_file in tqdm(zip(clean_files, noisy_files, noise_info_files),
                                                            total=len(clean_files), desc=f'Processing {set} data'):
            # Load the files
            clean_ecg = np.load(os.path.join(clean_folder, clean_file))
            noisy_ecg = np.load(os.path.join(noisy_folder, noisy_file))
            noise_info = np.load(os.path.join(noise_info_path, noise_info_file), allow_pickle=True)

            snr_values = []

            if noise_info.size == 0:  # If no noise intervals are found
                snr_values.append(1)
            else:
                for noise_interval in noise_info:
                    ma, em, bw, start, end = noise_interval

                    # Select the segments from the clean and noisy signals
                    clean_segment = clean_ecg[start:end]
                    noisy_segment = noisy_ecg[start:end]

                    # Calculate the SNR for this segment
                    snr_value = snr(clean_segment, noisy_segment)

                    # Skip the file if the SNR calculation failed (due to division by zero or invalid SNR)
                    if snr_value is None or snr_value < 0:
                        skipped_files.append(clean_file)
                        break

                    # Scale the SNR value
                    scaled_snr_value = scale_snr(snr_value)
                    snr_values.append(scaled_snr_value)

            # Save the scaled SNR values if there were no skips
            if len(snr_values) == len(noise_info):
                output_file = os.path.join(set_output_folder, clean_file)
                np.save(output_file, snr_values)

        if skipped_files:
            print(f"Skipped {len(skipped_files)} files in {set} due to invalid SNR calculations.")
            report_path = os.path.join(output_folder, f'{set}_snr_skipped_report.txt')
            with open(report_path, 'w') as report_file:
                report_file.write(f"Skipped {len(skipped_files)} files due to invalid SNR calculations:\n")
                report_file.write("\n".join(skipped_files))
            print(f"Report for {set} saved to {report_path}")


base_clean = r'C:\Users\marci\paper_proj_dataset\ptb_xl_clean'
base_noisy = r'C:\Users\marci\paper_proj_dataset\ptb_xl_noisy'
noise_info_folder = os.path.join(base_noisy, 'noise_info')
output_folder = r'C:\Users\marci\paper_proj_dataset\snr_int'

os.makedirs(output_folder, exist_ok=True)

calculate_global_snr_for_folder(base_clean, base_noisy, noise_info_folder, output_folder)
