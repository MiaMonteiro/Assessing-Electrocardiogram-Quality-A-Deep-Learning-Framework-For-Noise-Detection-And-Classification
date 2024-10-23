import numpy as np
import os
import neurokit2 as nk
import importlib.util
import sys
import pandas as pd
from tqdm import tqdm

# noisy_folder = r'C:\Users\marci\Proj_Tese\ptb_xl_360\test_noisy_order_fixed'
# clean_folder = r'C:\Users\marci\Proj_Tese\ptb_xl_360\test_clean_order_fixed'
#
# # the signals to be tested are:
# signal_names = ['19.npy', '12540.npy', '12787.npy', '4074.npy', '4983.npy', '8641.npy', '5574.npy', '6575.npy']
#
# # + 19: BW isolated
# # - 12540: EM, MA isolated
# # + 12787: EM + BW , MA + BW
# # +/- 4074: MA + EM
# # + 4983: MA + EM innacurate
# # - 8641: MA + EM innacurate
# # - 55747: MA + EM + BW
# # + 6575: MA + EM + BW innacurate
#
# # load the signals
#
# noisy_signals = {}
# clean_signals = {}
#
# for signal_name in signal_names:
#     # Construct full file paths
#     noisy_file_path = os.path.join(noisy_folder, signal_name)
#     clean_file_path = os.path.join(clean_folder, signal_name)
#
#     # Load the signals using numpy
#     noisy_signals[signal_name] = np.load(noisy_file_path)
#     clean_signals[signal_name] = np.load(clean_file_path)
#     # example of usage
#     # noisy_signals['19.npy']
#     # clean_signals['19.npy']
#
# # now we will apply traditional methods to these signals
#
# # FIRST NEUROKIT2's Zhao method
# zhao2018 = {}
#
# # Iterate over the noisy signals and check quality
# for signal_name, ecg_noisy in noisy_signals.items():
#     # Apply the neurokit2 ecg_quality function
#     quality = nk.ecg_quality(ecg_noisy, sampling_rate=360, method='zhao2018')
#     zhao2018[signal_name] = quality
# # QRS wave power spectrum distribution pSQI, kurtosis kSQI, and baseline relative power basSQI.
#
# # LIST OF SQIS
# Define the path to sqis.py
sqis_path = r'C:\Users\marci\signal_quality\signal_quality\sqis.py'  # Adjust this path

# Load the module
spec = importlib.util.spec_from_file_location("sqis", sqis_path)
sqis = importlib.util.module_from_spec(spec)
sys.modules["sqis"] = sqis
spec.loader.exec_module(sqis)
#
# # Now you can use sqis functions
#
# def calculate_snr_power_based(clean_signal, noisy_signal):
#     # Calculate signal power and noise power
#     signal_power = np.mean(np.square(clean_signal))
#     noise_power = np.mean(np.square(noisy_signal - clean_signal))
#
#     if noise_power == 0:
#         return np.inf, np.inf  # Return infinity if there's no noise
#
#     # Calculate dimensionless SNR (power ratio)
#     snr_adimensional = signal_power / noise_power
#
#     # Calculate SNR in dB
#     snr_db = 10 * np.log10(snr_adimensional)
#
#     return snr_db
#
# # Iterate over the noisy signals and check quality
# import neurokit2 as nk
#
# # Initialize a dictionary to store all the results
# results_list = []
#
# # Iterate over the signal names directly
# for signal_name in signal_names:
#     # Fetch the clean and noisy signals
#     clean_signal = clean_signals[signal_name]
#     noisy_signal = noisy_signals[signal_name]
#
#     # # Apply the neurokit2 ecg_peaks function on the clean signal
#     # peaks, info = nk.ecg_peaks(clean_signal, sampling_rate=360)
#     #
#     # peaks = list(map(int, info['ECG_R_Peaks']))  # Convert peaks to a list of integers
#
#     # Apply the sqis get_ecg_sqis function on the noisy signal
#     kurt = sqis.k_sqi(noisy_signal, kurtosis_method='fisher')
#     kurt_clean = sqis.k_sqi(clean_signal, kurtosis_method = 'fisher')
#
#     skew = sqis.s_sqi(noisy_signal)
#     skew_clean = sqis.s_sqi(clean_signal)
#
#     snr = calculate_snr_power_based(clean_signal, noisy_signal)
#
#     psd = sqis.p_sqi(noisy_signal, 360, 5)
#     psd_clean = sqis.p_sqi(clean_signal, 360, 5)
#
#     bas = sqis.bas_sqi(noisy_signal, 360, 5)
#     bas_clean = sqis.bas_sqi(clean_signal, 360, 5)
#
#     result_dict = {
#         'signal_name': signal_name,
#         'zhao2018': zhao2018[signal_name],
#         'kurtosis': kurt,
#         'kurtosis_clean': kurt_clean,
#         'skewness': skew,
#         'skewness_clean': skew_clean,
#         'snr': snr,
#         # 'snr_clean': snr_clean,
#         'p': psd,
#         'p_clean': psd_clean,
#         'bas': bas,
#         'bas_clean': bas_clean
#     }
#
#     # Append the result dictionary to the results list
#     results_list.append(result_dict)
#
# # Convert the results list into a DataFrame
# df_results = pd.DataFrame(results_list)
# df_results = df_results.round(3)
# # Save the DataFrame to an Excel file
# df_results.to_excel('ecg_sqi_results_updated.xlsx', index=False)
#
# print('Results saved to ecg_sqi_results.xlsx')

######### PART 2 ##########

# Define the folders
noisy_folder = r'C:\Users\marci\Proj_Tese\ptb_xl_360\test_noisy_order_fixed'
clean_folder = r'C:\Users\marci\Proj_Tese\ptb_xl_360\test_clean_order_fixed'

# Initialize dictionaries
noisy_signals = {}
clean_signals = {}

# Load noisy signals
# Load noisy signals with progress tracking
print("Loading noisy signals...")
for file_name in tqdm(os.listdir(noisy_folder), desc="Noisy signals"):
    if file_name.endswith('.npy'):
        # Create full file path
        file_path = os.path.join(noisy_folder, file_name)
        # Load the numpy array
        noisy_signals[file_name] = np.load(file_path)

# Load clean signals
print("Loading clean signals...")
for file_name in tqdm(os.listdir(clean_folder), desc="Clean signals"):
    if file_name.endswith('.npy'):
        # Create full file path
        file_path = os.path.join(clean_folder, file_name)
        # Load the numpy array
        clean_signals[file_name] = np.load(file_path)

def calculate_snr_power_based(clean_signal, noisy_signal):
    # Implement the SNR calculation
    power_clean = np.sum(clean_signal ** 2)
    power_noise = np.sum((noisy_signal - clean_signal) ** 2)
    return 10 * np.log10(power_clean / power_noise)


# Initialize lists to store metric data
metrics_data = {
    'kurt': [],
    'kurt_clean': [],
    'skew': [],
    'skew_clean': [],
    'snr': [],
    'psd': [],
    'psd_clean': [],
    'bas': [],
    'bas_clean': []
}


kurt_fails, psd_fails, bas_fails, skew_fails, snr_fails = 0, 0, 0, 0, 0
kurt_corrects, psd_corrects, bas_corrects, skew_corrects, snr_corrects = 0, 0, 0, 0, 0
clean_count = 0
noisy_count = 0

keys_to_remove = []

keys_to_remove = []

# Loop over all signals
print("Processing signals...")
for file_name in tqdm(clean_signals.keys(), desc="Signals"):
    # Retrieve the clean and noisy signals
    clean_signal = clean_signals[file_name]
    noisy_signal = noisy_signals[file_name]

    # Calculate SNR
    snr = calculate_snr_power_based(clean_signal, noisy_signal)

    # Check if SNR is infinite and exclude the signal if true
    if np.isinf(snr):
        keys_to_remove.append(file_name)  # Mark the file for removal
        continue  # Skip the rest of the loop for this file

    # If SNR is valid, proceed with processing
    clean_count += 1
    noisy_count += 1

    kurt_noisy = sqis.k_sqi(noisy_signal, kurtosis_method='fisher')
    kurt_clean = sqis.k_sqi(clean_signal, kurtosis_method='fisher')

    # Apply conditions and update counters for kurtosis
    if kurt_clean <= 5:
        kurt_fails += 1
    else:
        kurt_corrects += 1

    if kurt_noisy > 5:
        kurt_fails += 1
    else:
        kurt_corrects += 1

    # Calculate and update PSD
    psd_noisy = sqis.p_sqi(noisy_signal, 360, 5)
    psd_clean = sqis.p_sqi(clean_signal, 360, 5)

    if psd_clean <= 0.9:
        psd_fails += 1
    else:
        psd_corrects += 1

    if psd_noisy > 0.9:
        psd_fails += 1
    else:
        psd_corrects += 1

    # Calculate and update BAS
    bas_noisy = sqis.bas_sqi(noisy_signal, 360, 5)
    bas_clean = sqis.bas_sqi(clean_signal, 360, 5)

    if bas_clean <= 0.95:
        bas_fails += 1
    else:
        bas_corrects += 1

    if bas_noisy > 0.95:
        bas_fails += 1
    else:
        bas_corrects += 1

    # Calculate and update Skew
    skew_noisy = sqis.s_sqi(noisy_signal)
    skew_clean = sqis.s_sqi(clean_signal)

    if skew_clean > 0.8:
        skew_fails += 1
    else:
        skew_corrects += 1  # This includes the case when skew_clean <= 0.8

    if -0.8 <= skew_noisy <= 0.8:
        skew_fails += 1
    if -0.8 <= skew_clean <= 0.8:
        skew_corrects += 1

    if skew_noisy > 0.8:
        skew_corrects += 1
    if skew_noisy <= -0.8:
        skew_corrects += 1

    # Apply conditions and update counters for SNR
    if snr > 10:
        snr_corrects += 1
    else:
        snr_fails += 1

    # Append metrics to the lists
    metrics_data['kurt'].append(kurt_noisy)
    metrics_data['kurt_clean'].append(kurt_clean)
    metrics_data['skew'].append(skew_noisy)
    metrics_data['skew_clean'].append(skew_clean)
    metrics_data['snr'].append(snr)
    metrics_data['psd'].append(psd_noisy)
    metrics_data['psd_clean'].append(psd_clean)
    metrics_data['bas'].append(bas_noisy)
    metrics_data['bas_clean'].append(bas_clean)

# Remove invalid signals after processing
for key in keys_to_remove:
    del clean_signals[key]
    del noisy_signals[key]

print("Clean count:", clean_count)
print("Noisy count:", noisy_count)

# Calculate min, max, mean for each metric
metrics_summary = {}
for metric, values in metrics_data.items():
    if len(values) > 0:  # Only calculate if there are values
        metrics_summary[metric] = {
            'min': np.min(values),
            'max': np.max(values),
            'mean': np.mean(values),
            'std': np.std(values, ddof=0)
        }
    else:
        metrics_summary[metric] = {
            'min': None,
            'max': None,
            'mean': None,
            'std': None
        }

# Convert metrics_summary to a DataFrame for summary statistics
summary_stats_df = pd.DataFrame.from_dict(metrics_summary, orient='index')

# Save the summary statistics to an Excel file
summary_stats_file = 'metrics_calculated.xlsx'
summary_stats_df.to_excel(summary_stats_file, index=True)
print(f"Metrics summary statistics saved to {summary_stats_file}")

# Prepare a new DataFrame for percentages
percentages_summary = {
    'Metric': [],
    'Fail Percentage': [],
    'Correct Percentage': []
}

# Add percentages for each metric
total_signals = clean_count + noisy_count

# Add percentages for each metric
for metric in metrics_summary.keys():
    if metric == 'snr':
        percentages_summary['Metric'].append(metric)
        percentages_summary['Fail Percentage'].append((snr_fails / clean_count) * 100 if clean_count > 0 else 0)
        percentages_summary['Correct Percentage'].append((snr_corrects / clean_count) * 100 if clean_count > 0 else 0)
    else:
        percentages_summary['Metric'].append(metric)

        # Using the correct fail and correct variable names directly
        fail_var = f"{metric}_fails"
        correct_var = f"{metric}_corrects"

        if fail_var in locals() and correct_var in locals():
            percentages_summary['Fail Percentage'].append(
                (locals()[fail_var] / total_signals) * 100 if total_signals > 0 else 0)
            percentages_summary['Correct Percentage'].append(
                (locals()[correct_var] / total_signals) * 100 if total_signals > 0 else 0)
        else:
            percentages_summary['Fail Percentage'].append(0)  # Default to 0 if not found
            percentages_summary['Correct Percentage'].append(0)  # Default to 0 if not found

# Create a DataFrame for percentages
percentages_df = pd.DataFrame(percentages_summary)

# Save the percentages to a separate Excel file
percentages_file = 'metrics_summary_percentages.xlsx'
percentages_df.to_excel(percentages_file, index=False)
print(f"Metrics summary percentages saved to {percentages_file}")