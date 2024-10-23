import numpy as np
from sklearn.preprocessing import minmax_scale
import os
import random
from fitlers import smooth
from tqdm import tqdm
from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, resample
from scipy.stats import zscore
from sklearn.preprocessing import minmax_scale
from fitlers import filtering
from tqdm import tqdm

def sort_leads(dict):
    return dict[1]

ecg_12 = np.load(r'C:\Users\marci\paper_proj_dataset\ptb_xl_360_split\x_test_all_leads\0.npy')

ecg= np.zeros((3600,12))
for i in range(12):
    ecg[:,i] = resample(ecg_12[:,i], 3600)
    ecg[:,i] = zscore(ecg[:,i])
    ecg[:,i] = filtering(ecg[:,i])



print(ecg_12.shape)

plt.plot(ecg_12[:,0])
plt.title(f"raw 12 L")
plt.show()
record_info = []

num_leads = ecg_12.shape[1]

rpeaks_dict = {}

lead_rpeaks_count = []
all_peaks_dict = {}

lead_all_peaks_count = []
for lead_idx in range(num_leads):
    signal = ecg[:, lead_idx]

    # signal_resampled = resample(signal, 3600)
    # # print("resample",signal_resampled.shape)
    # # plt.plot(signal_resampled)
    # # plt.title(f"resampled")
    # # plt.show()
    # signal_zscore = zscore(signal_resampled)
    # # plt.plot(signal_zscore)
    # # plt.title(f"zscore")
    # # plt.show()
    # signal_filtered = filtering(signal_zscore)
    # plt.plot(signal_filtered)
    # plt.title(f"filtered")
    # plt.show()
    # signal_minmax = minmax_scale(signal_zscore)

    rpeaks, _ = find_peaks(signal, distance=300)  # r peaks
    all_peaks, _ = find_peaks(signal)    # print('all peaks for lead: ',lead_idx, ' - ', len(all_peaks))

    # Guardar os peaks para cada lead
    rpeaks_dict[lead_idx] = rpeaks
    # Guardar o numero de peaks para cada lead
    lead_rpeaks_count.append((lead_idx, len(rpeaks)))

    # guardar all peaks para cada lead
    all_peaks_dict[lead_idx] = all_peaks
    lead_all_peaks_count.append((lead_idx, len(all_peaks)))

# print('######')
# print(f"lead_rpeaks_count: {lead_rpeaks_count}")
# print(f"lead_all_peaks_count: {lead_all_peaks_count}")

# Filter leads com o min_peaks
filtered_leads = [lead for lead, count in lead_rpeaks_count if count >= 8]
# print(f"filtered_leads: {filtered_leads}")
dict_filtered = [lead_all_peaks_count[key] for key in filtered_leads]
# print(f"dict_filtered: {dict_filtered}")
# Sort leads by number of peaks
dict_filtered.sort(key=sort_leads)
# print(f"dict_filtered after sort: {dict_filtered}")

lowest_leads = dict_filtered[:3]  # busca as 3 leads com menos peaks
# print(f"lowest_leads: {lowest_leads}")

idx_lowest = [i for i, j in lowest_leads]
# print(idx_lowest)
# Save the signals for the selected leads
  # Save the signal corresponding to this lead
signal = ecg[:, lead_idx]
    # signal_resampled = resample(signal, 3600)
    # signal_minmax = minmax_scale(signal_resampled)
    # signal_to_save = filtering(signal)

print('clean shape', signal.shape)
plt.plot(signal)

plt.title('clean_final')
plt.show()

################ 4 ##########3

from sklearn.model_selection import train_test_split
import numpy as np
import os
from tools.process_noise import process_noise
from tools.add_noise import add_noise_to_signals_in_folder

# Sampling rate (Hz)
sampling_rate = 360
total_length = 3600

# Access the noise data from the pickle file generated in 1_get_records.py
dir_noise = r'/mit_bih_noise_stress'
bw_raw = np.load(os.path.join(dir_noise, 'bw.npy'))
ma_raw = np.load(os.path.join(dir_noise, 'ma.npy'))
em_raw = np.load(os.path.join(dir_noise, 'em.npy'))
# Process each noise type, apply the function to the last 3 elements of the noise_in list
print(bw_raw.shape)
plt.plot(bw_raw[:3600,0])
plt.title('bw_raw')
plt.show()
bw_processed = process_noise(bw_raw)
em_processed = process_noise(em_raw)
ma_processed = process_noise(ma_raw)

plt.plot(bw_processed[:3600])
plt.title('bw_processed')
plt.show()

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
total_samples = len(signal)
plt.plot(noise_data_train['BW'][:3600])
plt.title('noise_data_train')
plt.show()
# Extract the types of noise available from the noise_data dictionary
noise_types = list(noise_data_train.keys())
max_intervals = 4
# Randomly determine the number of noise intervals to add, with a maximum of max_intervals
num_intervals = np.random.randint(0, max_intervals + 1)
# +1 para incluir o 4 senao 0 1 2 3

# num_intervals = 4
# If no intervals are selected, return the original signal and an empty noise_info list
if num_intervals != 0:

# Make a copy of the original signal to modify without altering the original
    signal_copy = signal.copy().flatten()
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
        start_noise = np.random.randint(0, len(noise_data_train[selected_noise]) - interval_length)
        noise = noise_data_train[selected_noise][start_noise:start_noise + interval_length]

        # Add the selected noise to the specified interval in the combined_noise array
        combined_noise[start_time:end_time] = noise

        plt.plot(combined_noise)
        plt.title('combined_noise for interval ')
        plt.show()

        plt.plot(noise)
        plt.title('noise for interval ')
        plt.show()
        # print('start time', start_time)
        # print('end time', end_time)
        smoothing_window = 70
        if start_time > 35:
            smoothed_transitions_noise_start = smooth(combined_noise[start_time - 35:start_time + 35], smoothing_window)
            # print(f"smoothed_transitions_noise_start shape: {smoothed_transitions_noise_start.shape}")
            combined_noise[start_time - 35:start_time + 35] = smoothed_transitions_noise_start
            # print(f"combined_noise[start_time-35:start_time+35] shape: {combined_noise[start_time-35:start_time+35].shape}")
        else:
            smoothing_window_adapt = start_time + 35
            smoothed_transitions_noise_start = smooth(combined_noise[0:start_time + 35], smoothing_window_adapt)
            # print(f"smoothed_transitions_noise_start shape: {smoothed_transitions_noise_start.shape}")
            combined_noise[0:start_time + 35] = smoothed_transitions_noise_start
            # print(f"combined_noise[0:start_time+35] shape: {combined_noise[0:start_time+35].shape}")

        if end_time < total_samples - 35:
            smoothed_transitions_noise_end = smooth(combined_noise[end_time - 35:end_time + 35], smoothing_window)
            # print(f"smoothed_transitions_noise_end shape: {smoothed_transitions_noise_end.shape}")
            combined_noise[end_time - 35:end_time + 35] = smoothed_transitions_noise_end
            # print(f"combined_noise[end_time-35:end_time+35] shape: {combined_noise[end_time-35:end_time+35].shape}")

        else:
            smoothing_window_adapt = total_samples - end_time + 35
            # print(f"smoothing_window: {smoothing_window}")
            smoothed_transitions_noise_end = smooth(combined_noise[end_time - 35:total_samples], smoothing_window_adapt)
            # print(f"smoothed_transitions_noise_end shape: {smoothed_transitions_noise_end.shape}")
            combined_noise[end_time - 35:total_samples] = smoothed_transitions_noise_end
            # print(f"combined_noise[end_time-35:total_samples] shape: {combined_noise[end_time-35:total_samples].shape}")

        # Record the selected noise type in the active_noise_types array
        active_noise_types = [0] * len(noise_types)
        active_noise_types[noise_types.index(selected_noise)] = 1

        # Combine the interval information and active noise type into a single entry
        interval_info = [start_time, end_time] + active_noise_types
        noise_info.append(interval_info)

    # scaled_clean_signal = minmax_scale(signal)  # Scale clean signal between -1 and 1
    # scaled_combined_noise = minmax_scale(combined_noise)  # Scale noise between -1 and 1

    # Scale the noise before adding it to the signal
    # scale_factor = random.uniform(0.1, 0.8)
    # noisy_signal = scaled_clean_signal + scale_factor * scaled_combined_noise

    # # Scale the smoothed noise and add it to the signal
    scale_factor = random.uniform(0.1, 0.8)
    print (f"scale factor: {scale_factor}")
    signal_copy += scale_factor * combined_noise

plt.plot(signal_copy)
plt.title('signal_noisy')
plt.show()
