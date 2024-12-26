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


# def process_signals( records_folder, output_folder, min_peaks=8, peak_distance=300):
#     # Create output directory if it does not exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     # Initialize counters
#     total_files = len([f for f in os.listdir(records_folder) if f.endswith(".npy")])
#     total_leads = total_files * 12
#     leads_saved = 0
#
#     record_info = []
#
#
#
#     with tqdm(total=total_files, desc='Processing Files') as file_pbar:
#         for file_name in os.listdir(records_folder):
#             if file_name.endswith(".npy"):
#
#                 file_path = os.path.join(records_folder, file_name)
#                 signals = np.load(file_path)
#                 num_leads = signals.shape[1]
#
#                 rpeaks_dict = {}
#
#                 lead_rpeaks_count = []
#                 all_peaks_dict = {}
#
#                 lead_all_peaks_count = []
#                 for lead_idx in range(num_leads):
#                     signal = signals[:, lead_idx]
#
#                     signal_resampled = resample(signal, 3600)
#                     signal_zscore = zscore(signal_resampled)
#                     signal_filtered = filtering(signal_zscore)
#                     # signal_minmax = minmax_scale(signal_zscore)
#
#                     rpeaks, _ = find_peaks(signal_filtered, distance=peak_distance) # r peaks
#                     all_peaks, _ = find_peaks(signal_filtered)
#                     # print('all peaks for lead: ',lead_idx, ' - ', len(all_peaks))
#
#                     # Guardar os peaks para cada lead
#                     rpeaks_dict[lead_idx] = rpeaks
#                     # Guardar o numero de peaks para cada lead
#                     lead_rpeaks_count.append((lead_idx, len(rpeaks)))
#
#                     #guardar all peaks para cada lead
#                     all_peaks_dict[lead_idx] = all_peaks
#                     lead_all_peaks_count.append((lead_idx, len(all_peaks)))
#
#
#                 # print('######')
#                 # print(f"lead_rpeaks_count: {lead_rpeaks_count}")
#                 # print(f"lead_all_peaks_count: {lead_all_peaks_count}")
#
#                 # Filter leads com o min_peaks
#                 filtered_leads = [lead for lead, count in lead_rpeaks_count if count >= min_peaks]
#                 # print(f"filtered_leads: {filtered_leads}")
#                 dict_filtered = [lead_all_peaks_count[key] for key in filtered_leads]
#                 # print(f"dict_filtered: {dict_filtered}")
#                 # Sort leads by number of peaks
#                 dict_filtered.sort(key=sort_leads)
#                 # print(f"dict_filtered after sort: {dict_filtered}")
#
#                 lowest_leads = dict_filtered[:3]  # busca as 3 leads com menos peaks
#                 # print(f"lowest_leads: {lowest_leads}")
#
#                 idx_lowest = [i for i, j in lowest_leads]
#                 # print(idx_lowest)
#                 # Save the signals for the selected leads
#                 for idx, lead_idx in enumerate(idx_lowest):
#                     suffix = idx + 1  # Create suffix (1 for the lead with the fewest peaks, 2 for the second fewest, etc.)
#                     output_file_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_{suffix}.npy")
#
#                     # Save the signal corresponding to this lead
#                     signal = signals[:, lead_idx]
#                     # signal_resampled = resample(signal, 3600)
#                     # signal_minmax = minmax_scale(signal_resampled)
#                     # signal_to_save = filtering(signal)
#
#                     with open(output_file_path, 'wb') as f:
#                         np.save(f, signal)
#                         leads_saved += 1
#
#                 # Track saved and dropped leads
#                 # Extract the ECG ID from the file name
#                 ecg_id = int(
#                     os.path.splitext(file_name)[0])  # Assumes file name is just the ID without the extension
#
#                 record_info.append({
#                     'ecg_id': ecg_id,
#                     'saved_leads': str(idx_lowest),
#                     'dropped_leads': str(list(set(range(num_leads)) - set(idx_lowest)))
#                 })
#
#                 file_pbar.update(1)  # Update progress bar after processing each file
#
#         # Print summary after processing all files
#         print(f"Total leads processed: {total_leads}")
#         print(f"Total leads saved: {leads_saved}")
#
#     return record_info

def process_signals(records_folder, output_folder, min_peaks=8, peak_distance=300):
    # Create output directory if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize counters
    total_files = len([f for f in os.listdir(records_folder) if f.endswith(".npy")])
    total_leads = total_files * 12
    leads_saved = 0

    record_info = []

    with tqdm(total=total_files, desc='Processing Files') as file_pbar:
        for file_name in os.listdir(records_folder):
            if file_name.endswith(".npy"):

                file_path = os.path.join(records_folder, file_name)
                ecg_12 = np.load(file_path)
                ecg = np.zeros((3600,12))
                for i in range(12):
                    ecg[:, i] = resample(ecg_12[:, i], 3600)
                    ecg[:, i] = zscore(ecg[:, i])
                    ecg[:, i] = filtering(ecg[:, i])

                rpeaks_dict = {}

                lead_rpeaks_count = []
                all_peaks_dict = {}

                lead_all_peaks_count = []
                for lead_idx in range(12):
                    signal = ecg[:, lead_idx]

                    rpeaks, _ = find_peaks(signal, distance=peak_distance)  # r peaks
                    all_peaks, _ = find_peaks(signal)
                    # print('all peaks for lead: ',lead_idx, ' - ', len(all_peaks))

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
                filtered_leads = [lead for lead, count in lead_rpeaks_count if count >= min_peaks]
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
                for idx, lead_idx in enumerate(idx_lowest):
                    suffix = idx + 1  # Create suffix (1 for the lead with the fewest peaks, 2 for the second fewest, etc.)
                    output_file_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_{suffix}.npy")

                    # Save the signal corresponding to this lead
                    signal = ecg[:, lead_idx]
                    # signal_resampled = resample(signal, 3600)
                    # signal_minmax = minmax_scale(signal_resampled)
                    # signal_to_save = filtering(signal)

                    with open(output_file_path, 'wb') as f:
                        np.save(f, signal)
                        leads_saved += 1

                # Track saved and dropped leads
                # Extract the ECG ID from the file name
                ecg_id = int(
                    os.path.splitext(file_name)[0])  # Assumes file name is just the ID without the extension

                record_info.append({
                    'ecg_id': ecg_id,
                    'saved_leads': str(idx_lowest),
                    'dropped_leads': str(list(set(range(12)) - set(idx_lowest)))
                })

                file_pbar.update(1)  # Update progress bar after processing each file

        # Print summary after processing all files
        print(f"Total leads processed: {total_leads}")
        print(f"Total leads saved: {leads_saved}")

    return record_info

def track_and_select_clean_leads(csv_path, records_folder, output_folder, output_csv_path):
    original_df = pd.read_csv(csv_path)

    record_info = process_signals(records_folder, output_folder)
    record_df = pd.DataFrame(record_info)

    # Merge on 'ecg_id'
    merged_df = original_df.merge(record_df, on='ecg_id', how='left')

    # Drop rows where saved_leads is NaN (i.e., not in the current set)
    cleaned_df = merged_df.dropna(subset=['saved_leads'])

    cleaned_df.to_csv(output_csv_path, index=False)
    print(f"Processed CSV saved to: {output_csv_path}")



# train_dir = os.path.join(split_dir, 'train_all_leads1')
# test_dir = os.path.join(split_dir, 'test_all_leads1')
# val_dir = os.path.join(split_dir, 'val_all_leads1')
#
# processed_dir = r'C:\Users\marci\Proj_Tese\ptb_xl_360_1'
# os.makedirs(processed_dir, exist_ok=True)
#
# # Process signals for each set
# process_signals(train_dir, os.path.join(processed_dir, 'train_selected_leads'))
# process_signals(test_dir, os.path.join(processed_dir, 'test_selected_leads'), )
# process_signals(val_dir, os.path.join(processed_dir, 'val_selected_leads'))

