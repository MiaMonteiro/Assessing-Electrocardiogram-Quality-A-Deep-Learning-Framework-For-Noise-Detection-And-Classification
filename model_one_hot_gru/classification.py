import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl

# the idea is to take a prediction and produce a report of the classification
# 1,3600,3
# add a option if the SNR power is not available

# snr
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
#     return snr_adimensional, snr_db
#
#
# # Function to calculate SNR using mean and standard deviation (dimensionless)
# def calculate_snr_mean_std(signal):
#     mu = np.mean(signal)  # Mean of the signal
#     sigma = np.std(signal)  # Standard deviation of the signal
#
#     if sigma == 0:
#         return np.inf  # Return infinity if there's no variation in the signal
#
#     # SNR as the ratio of mean to standard deviation
#     snr_adimensional = mu / sigma
#     return snr_adimensional

def classify(pred):

    num_timesteps, num_classes = pred.shape
    ma_count, em_count, bw_count = 0, 0, 0

    intervals = {'MA': [], 'EM': [], 'BW': []}
    time_intervals = {'MA': [], 'EM': [], 'BW': []}

    # snr_power_adimensional, snr_power_db = calculate_snr_power_based(clean_signal, noisy_signal)
    # snr_mean_std = calculate_snr_mean_std(clean_signal)
    #
    # # If SNR is infinite, replace with a more representative string
    # snr_power_message = "The noisy signal is identical to the clean signal meaning that there's no detectable noise" if snr_power_adimensional == np.inf else f"{snr_power_adimensional:.2f}"
    # snr_db_message = "The noisy signal is identical to the clean signal meaning that there's no detectable noise)" if snr_power_db == np.inf else f"{snr_power_db:.2f} dB"

    # find if any noise is present
    noise_active = np.any(pred > 0, axis=1)

    # find the % of the signal filled with noise
    noise_pcent = round((np.sum(noise_active)/num_timesteps) * 100,2)

    for i in range(num_timesteps):  # Iterate over each time step
        if pred[i, 0] == 1:
            ma_count += 1
        if pred[i, 1] == 1:
            em_count += 1
        if pred[i, 2] == 1:
            bw_count += 1

    # print(ma_count, em_count, bw_count)
    # find the % of the signal filled with each class of noise
    ma_pcent = round((ma_count / num_timesteps) * 100,2)
    em_pcent = round((em_count / num_timesteps) * 100, 2)
    bw_pcent = round((bw_count / num_timesteps) * 100, 2)

    if noise_pcent == 0:
        quality = f'\033[1;32mThe signal is perfectly clean, with 0% noise contamination\033[0m'
    elif noise_pcent < 10:
        quality = f'\033[1;92mLess than 10% of the signal is contaminated with noise\033[0m'  # Bright Green (Bold)
        # quality = f'Excellent'
    elif noise_pcent < 25:
        quality = f'\033[1m\033[93mLess than 25% of the signal is contaminated with noise\033[0m'    # Yellow
        # quality = f'Minor Noise'
    elif noise_pcent < 40:
        quality = f'\033[1m\033[38;5;208mLess than 40% of the signal is contaminated with noise\033[0m'  # Orange (different shade)
        # quality = f'Noticeable Noise'
    else:
        quality = f'\033[1m\033[91mMore than 40% of the signal is contaminated with noise\033[0m'  # Red
        # quality = f'High Noise'

    # find the duration of each class of noise active

    for i, class_name in enumerate(['MA', 'EM', 'BW']):
        noise_indices = np.where(pred[:, i] > 0)[0]

        if len(noise_indices) > 0:
            start = noise_indices[0]
            end = start
            for j in range(1, len(noise_indices)):
                if noise_indices[j] - noise_indices[j - 1] <= 360:
                    end = noise_indices[j]
                else:
                    intervals[class_name].append((start, end))
                    start = noise_indices[j]
                    end = start
            intervals[class_name].append((start, end))

        # for class_name in intervals:

        for class_name in intervals:
            time_intervals[class_name] = [
                ((start * 10) / 3600,
                 (end * 10) / 3600)
                for start, end in intervals[class_name]
            ]

        for class_name in time_intervals:
            time_intervals[class_name] = [(round(start, 2), round(end, 2)) for start, end in time_intervals[class_name]]

        # Prepare the formatted report string
    report = (
        f"Noise Percentage: {noise_pcent}%\n"
        f"MA Percentage: {ma_pcent}%\n"
        f"EM Percentage: {em_pcent}%\n"
        f"BW Percentage: {bw_pcent}%\n"
        f"Quality: {quality}\n"
        f"MA Intervals (s): {time_intervals['MA']}\n"
        f"EM Intervals (s): {time_intervals['EM']}\n"
        f"BW Intervals (s): {time_intervals['BW']}"
    )

    # Return both the raw data and the formatted report
    return {
        'noise_percentage': noise_pcent,
        'percentage_ma': ma_pcent,
        'percentage_em': em_pcent,
        'percentage_bw': bw_pcent,
        'quality': quality,
        'intervals_ma_seconds': time_intervals['MA'],
        'intervals_em_seconds': time_intervals['EM'],
        'intervals_bw_seconds': time_intervals['BW'],
        # 'snr_power_adimensional': snr_power_adimensional,
        # 'snr_power_db': snr_power_db,
        # 'snr_mean_std': snr_mean_std,
        'report': report  # Add the formatted report string
    }
    # # Print the result dictionary in a more readable format
    # for key, value in result.items():
    #     print(f'{key}: {value}')

dir = r'C:\Users\marci\paper_proj_dataset\paper\ONEHOT_GRU_layers3_hiddensize_128_states3_patience40_biderectionalTrue_drop0.3_date2810_0950'

pred = np.load(dir + r'\preds5000.npy')

indx = np.load(dir + r'\idx_samples_to_save5000.npy')


results_list = []

# only the first 30
for i in range(500):
    result = classify(pred[i])
    print(f"\033[1mClassification for the signal {i + 1} of the test set:\033[0m")
    print()
    print(result['report'])  # Print the formatted report
    print('-' * 67)  # Prints a line with 50 dashes
    print("\n")


    # results_list.append({
    #     'Signal': i + 1,
    #     'Quality': result['quality'],
    #     'SNR (dB)': result['snr_power_db'],
    #     'SNR (Power, dimensionless)': result['snr_power_adimensional'],
    #     'SNR (Mean/Std, dimensionless)': result['snr_mean_std'],
    #     # 'Noise Percentage (%)': result['noise_percentage']
    # })

"""
High Noise Percentage Dominates: When the noise percentage is high, it outweighs other factors like SNR because even if the signal has good SNR, the sheer amount of noise will make it hard to reclassify to a lower category.

SNR Helps at Moderate Noise Levels: The ponderation system has the most impact on mid-range noise levels (30-50%), where SNR metrics can meaningfully contribute to downgrading a signal's noise classification. 
This is where the nuance of the ponderation system comes into play.

For Low and High Extremes: At low noise levels, the signals remain classified as "Excellent" because both the noise percentage and SNR indicate good signal quality. At high noise levels, 
even good SNR values aren't enough to move the classification because the overall signal is still heavily contaminated by noise.

"""

#     print(
#         f"Prediction {i + 1} - Quality: {result['quality']}, SNR (dB): {result['snr_power_db']:.2f}, SNR (Power, dimensionless): {result['snr_power_adimensional']:.2f}, SNR (Mean/Std): {result['snr_mean_std']:.2f}")

    # Add result to the list




    # df.to_csv('.\classification_results.csv', index=False)

#
#
# df = pd.DataFrame(results_list)
#
# # Display the DataFrame
# print("\nSNR Classification Results Table:")
# print(df)
#
# # Define a mapping from Quality strings to numeric values
# quality_mapping = {
#     'Excellent': 0,
#     'Minor Noise': 1,
#     'Noticeable Noise': 2,
#     'High Noise': 3
# }
#
# # Add a new column with numeric values for Quality
# df['Quality_Score'] = df['Quality'].map(quality_mapping)
#
# # Check the DataFrame structure to ensure conversion
# print(df.head())
#
# # Create a mask for rows where SNR (dB) and SNR (Power, dimensionless) are finite
# finite_mask = np.isfinite(df['SNR (dB)']) & np.isfinite(df['SNR (Power, dimensionless)'])
#
# # Correlation analysis only for finite SNR values in dB and Power
# correlations_finite = df[finite_mask][['SNR (dB)', 'SNR (Power, dimensionless)', 'SNR (Mean/Std, dimensionless)']].corrwith(df[finite_mask]['Quality_Score'])
#
# print("\nCorrelations with Quality Score (for finite SNR values):")
# print(correlations_finite)
#
# # Now, also calculate the correlation with SNR (Mean/Std, dimensionless) for all values (including inf)
# correlations_mean_std = df[['SNR (Mean/Std, dimensionless)']].corrwith(df['Quality_Score'])
# print("\nCorrelation of SNR (Mean/Std) with Quality Score (for all values):")
# print(correlations_mean_std)
#
#
# # Set the font to Palatino Linotype
# plt.rcParams['font.family'] = 'Palatino Linotype'
#
# # Define muted colors
# muted_green = '#7fbf7f'
# muted_red = '#f28e8e'
#
# # Set gray grid lines
# mpl.rcParams['grid.color'] = 'gray'
# mpl.rcParams['grid.linestyle'] = '--'
#
# # Your scatter plot code
# fig, axs = plt.subplots(2, 1, figsize=(10, 4))  # Adjust figure size as needed
#
# # First subplot: SNR (Power, dimensionless) vs Quality Score (only for finite values)
# axs[0].scatter(df[finite_mask]['SNR (Power, dimensionless)'], df[finite_mask]['Quality_Score'],
#                label='SNR (Power)', color=muted_green, alpha=0.5)
# axs[0].set_xlabel('SNR (Power, dimensionless)')
# axs[0].set_ylabel('Quality Score (0=Excellent, 3=High Noise)')
# axs[0].legend()
# axs[0].set_title('Relationship Between Quality Score and SNR (Power, dimensionless)')
# axs[0].grid(True)
#
# # Second subplot: SNR (Mean/Std, dimensionless) vs Quality Score (for all values)
# axs[1].scatter(df['SNR (Mean/Std, dimensionless)'], df['Quality_Score'],
#                label='SNR (Mean/Std)', color=muted_red, alpha=0.5)
# axs[1].set_xlabel('SNR (Mean/Std, dimensionless)')
# axs[1].set_ylabel('Quality Score (0=Excellent, 3=High Noise)')
# axs[1].legend()
# axs[1].set_title('Relationship Between Quality Score and SNR (Mean/Std, dimensionless)')
# axs[1].grid(True)
#
# # Adjust layout so subplots are more compact
# plt.subplots_adjust(hspace=0.4)  # Reduce space between subplots
#
# # Use tight_layout to automatically minimize whitespace
# plt.tight_layout()
#
# # Save the plot as an image file
# plt.savefig('SNR_vs_Quality_Score.png', format='png', dpi=300, bbox_inches='tight')  # Save at high resolution
#
# # Show the combined figure
# plt.show()



############ NO PONDERATION SYSTEM ####################

# if noise_pcent < 5:
#     # quality = f'\033[1;92mExcellent\033[0m'  # Bright Green (Bold)
#     quality = f'Excellent'
# elif noise_pcent < 15:
#     # quality = f'\033[93mMinor Noise\033[0m'  # Yellow
#     quality = f'Minor Noise'
# elif noise_pcent < 30:
#     # quality = f'\033[38;5;208mNoticeable Noise\033[0m'  # Orange (different shade)
#     quality = f'Noticeable Noise'
# else:
#     # quality = f'\033[91mHigh Noise\033[0m'  # Red
#     quality = f'High Noise'


# ==============================
# Add Ponderation System
# ==============================
#
# # Define weights for each factor
# weight_snr_mean_std = 0.1  # Lower weight due to weak correlation
# weight_snr_power = 0.4  # Lower weight due to weak correlation
# weight_noise_percentage = 0.5  # Higher weight for noise percentage
#
# # Contribution from SNR (Power, dimensionless)
# if np.isinf(snr_power_adimensional):  # If SNR Power is infinite, it's treated as a high-quality signal
#     snr_power_contribution = 1  # High-quality signal
# elif 100 <= snr_power_adimensional <= 500:
#     snr_power_contribution = 2  # Moderate contribution
# else:
#     snr_power_contribution = 3  # Strong indication of low quality
#
# # Contribution from SNR (Mean/Std)
# if snr_mean_std < 2:
#     snr_mean_std_contribution = 3  # Strong indication of low quality
# elif 2 <= snr_mean_std <= 5:
#     snr_mean_std_contribution = 2  # Moderate contribution
# else:
#     snr_mean_std_contribution = 1  # High-quality signal
#
# # Calculate the weighted score
# weighted_score = (
#         weight_noise_percentage * noise_pcent
#         + weight_snr_mean_std * snr_mean_std_contribution
#         + weight_snr_power * snr_power_contribution
# )
#
# # Use the weighted score to classify the signal
# if weighted_score < 5:
#     quality = f'\033[1;92mExcellent\033[0m'
#     # quality = f'Excellent'
# elif weighted_score < 15:
#     quality = f'\033[93mMinor Noise\033[0m'
#     # quality = f'Minor Noise'
# elif weighted_score < 30:
#     quality = f'\033[38;5;208mNoticeable Noise\033[0m'
#     # quality = f'Noticeable Noise'
# else:
#     quality = f'\033[91mHigh Noise\033[0m'
#     # quality = f'High Noise'
# # ==============================
# # End of Ponderation System
# # ==============================