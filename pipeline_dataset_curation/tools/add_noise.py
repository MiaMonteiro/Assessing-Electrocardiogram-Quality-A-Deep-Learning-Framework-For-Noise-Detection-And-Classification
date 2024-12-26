import numpy as np
from sklearn.preprocessing import minmax_scale
import os
import random
from fitlers import smooth
from tqdm import tqdm
from matplotlib import pyplot as plt

def add_noise_to_signal(clean_signal, noise_data, max_intervals=4):
    # Get the total number of samples in the signal
    total_samples = len(clean_signal)

    # Extract the types of noise available from the noise_data dictionary
    noise_types = list(noise_data.keys())

    # Randomly determine the number of noise intervals to add, with a maximum of max_intervals
    num_intervals = np.random.randint(0, max_intervals +1)
    # +1 para incluir o 4 senao 0 1 2 3

    #num_intervals = 4
    # If no intervals are selected, return the original signal and an empty noise_info list
    if num_intervals == 0:
        return clean_signal, []

    # Make a copy of the original signal to modify without altering the original
    signal = clean_signal.copy().flatten()
    # print(f"signal shape: {signal.shape}")
    # Initialize a list to store information about the noise intervals
    noise_info = []

    # Initialize an array to hold the combined noise for the interval
    combined_noise = np.zeros(total_samples)

    # Loop through each selected interval to add noise
    for i in range(num_intervals):
        # Randomly select the type of noise to inject
        selected_noise = np.random.choice(noise_types)

        if selected_noise == 'BW':
            # For BW noise, ensure that the interval length is between 1800 samples and the maximum possible length
            min_length = 1800
            start_time = np.random.randint(0, total_samples - min_length)
            max_length = total_samples - start_time
            interval_length = np.random.randint(min_length, max_length + 1)
        else:
            # For other types of noise, use a random interval length up to a maximum of 2000 samples
            interval_length = np.random.randint(360, min(2000, total_samples))
            start_time = np.random.randint(0, total_samples - interval_length)

        end_time = start_time + interval_length

        # Get the noise data for the selected interval
        start_noise = np.random.randint(0, len(noise_data[selected_noise]) - interval_length)
        noise = noise_data[selected_noise][start_noise:start_noise + interval_length]

        # Add the selected noise to the specified interval in the combined_noise array
        combined_noise[start_time:end_time] = noise

        # print('start time', start_time)
        # print('end time', end_time)
        smoothing_window = 70
        if start_time > 35:
            smoothed_transitions_noise_start = smooth(combined_noise[start_time-35:start_time+35], smoothing_window)
            # print(f"smoothed_transitions_noise_start shape: {smoothed_transitions_noise_start.shape}")
            combined_noise[start_time-35:start_time+35] = smoothed_transitions_noise_start
            # print(f"combined_noise[start_time-35:start_time+35] shape: {combined_noise[start_time-35:start_time+35].shape}")
        else:
            smoothing_window_adapt = start_time + 35
            smoothed_transitions_noise_start = smooth(combined_noise[0:start_time+35], smoothing_window_adapt)
            # print(f"smoothed_transitions_noise_start shape: {smoothed_transitions_noise_start.shape}")
            combined_noise[0:start_time+35] = smoothed_transitions_noise_start
            # print(f"combined_noise[0:start_time+35] shape: {combined_noise[0:start_time+35].shape}")

        if end_time < total_samples - 35:
            smoothed_transitions_noise_end = smooth(combined_noise[end_time-35:end_time+35], smoothing_window)
            # print(f"smoothed_transitions_noise_end shape: {smoothed_transitions_noise_end.shape}")
            combined_noise[end_time-35:end_time+35] = smoothed_transitions_noise_end
            # print(f"combined_noise[end_time-35:end_time+35] shape: {combined_noise[end_time-35:end_time+35].shape}")

        else:
            smoothing_window_adapt = total_samples - end_time + 35
            # print(f"smoothing_window: {smoothing_window}")
            smoothed_transitions_noise_end = smooth(combined_noise[end_time-35:total_samples], smoothing_window_adapt)
            # print(f"smoothed_transitions_noise_end shape: {smoothed_transitions_noise_end.shape}")
            combined_noise[end_time-35:total_samples] = smoothed_transitions_noise_end
            # print(f"combined_noise[end_time-35:total_samples] shape: {combined_noise[end_time-35:total_samples].shape}")

        # Record the selected noise type in the active_noise_types array
        active_noise_types = [0] * len(noise_types)
        active_noise_types[noise_types.index(selected_noise)] = 1

        # Combine the interval information and active noise type into a single entry
        interval_info = [start_time, end_time] + active_noise_types
        noise_info.append(interval_info)

    # Scale the smoothed noise and add it to the signal
    scale_factor = random.uniform(0.2, 1)
    signal += scale_factor * combined_noise

    return signal, noise_info


def add_noise_to_signals_in_folder(set_name, input_folder, output_base_folder, noise_data):
    # Determine output folders based on set_name
    output_folder = os.path.join(output_base_folder, f'{set_name}_noisy')
    noise_info_folder = os.path.join(output_base_folder, 'noise_info', f'{set_name}_noise_info')

    # Create the output and noise info directories if they don't exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(noise_info_folder, exist_ok=True)

    # List all files in the input folder
    files = os.listdir(input_folder)

    # Process each file in the folder
    for file_name in tqdm(files, desc=f"Processing files in {input_folder}"):
        if file_name.endswith('.npy'):
            # Load the original signal from the .npy file
            original_signal_path = os.path.join(input_folder, file_name)
            original_signal = np.load(original_signal_path)

            # Apply noise to the signal
            noisy_signal, noise_info = add_noise_to_signal(original_signal, noise_data)

            # Save the noisy signal as .npy file in the output folder
            output_signal_path = os.path.join(output_folder, file_name)
            np.save(output_signal_path, noisy_signal)

            # Save noise info as .npy file in the noise_info_folder
            noise_info_path = os.path.join(noise_info_folder, os.path.splitext(file_name)[0] + '_noise_info.npy')
            np.save(noise_info_path, np.array(noise_info, dtype=object))

