import os
import numpy as np
from matplotlib import pyplot as plt


##### Plot Noisy Signals and Original Signal ################################################################################################
def plot_signals(original_signal, transformed_signal, noise_info, plot_path, fs=360):
    plt.rcParams['font.family'] = 'Palatino Linotype'
    # Define colors for each combination (reuse from previous function)
    colors = {
        (1, 0, 0): '#ec5e42ce',  # Semi-transparent red (MA)
        (0, 1, 0): '#fcd51599',  # Yellowish with transparency (EM)
        (0, 0, 1): '#8eb0ddff',  # Opaque blue (BW)
        (1, 1, 0): '#f5b980ff',  # Opaque orange (MA + EM)
        (1, 0, 1): '#cf97e0ff',  # Opaque purple (MA + BW)
        (0, 1, 1): '#b2c993ff',  # Opaque greenish (EM + BW)
        (1, 1, 1): '#c5c5c5cc'   # Semi-transparent grey (MA + EM + BW)
    }

    # Convert sample index to time (seconds) using fs
    time = np.arange(len(original_signal)) / fs

    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot the original signal with time on x-axis
    axs[0].plot(time, original_signal, label='Original Signal', color='black', alpha=0.5)
    axs[0].set_ylabel('Amplitude', fontsize=26)
    axs[0].set_xlim(0, 10)  # Set x-axis limits to 0 to 10 seconds
    axs[0].set_yticks([0, 0.5, 1])  # Set y-axis ticks
    axs[0].tick_params(axis='both', which='major', labelsize=18)  # Set tick size
    # axs[0].grid(True)

    # Plot the transformed (noisy) signal with time on x-axis
    axs[1].plot(time, transformed_signal, label='Transformed Signal', color='black', alpha=0.5)
    axs[1].set_xlabel('Time (seconds)', fontsize=26)  # Set x-axis label with fontsize 26
    axs[1].set_ylabel('Amplitude', fontsize=26)  # Set y-axis label with fontsize 26
    axs[1].set_xlim(0, 10)  # Set x-axis limits to 0 to 10 seconds
    axs[1].set_yticks([0, 0.5, 1])  # Set y-axis ticks
    axs[1].tick_params(axis='both', which='major', labelsize=18)  # Set tick size
    # axs[1].grid(True)

    # Overlay noise intervals with vertical lines and shaded regions
    for interval in noise_info:
        start_idx = interval[0]
        end_idx = interval[1]
        key = tuple(interval[2:])  # Assuming the noise info includes the noise state as a tuple after the indices
        if key in colors:
            color = colors[key]

        # Convert sample indices to time
        start_time = start_idx / fs
        end_time = end_idx / fs

        # Shade the region in the transformed signal plot
        axs[1].axvspan(start_time, end_time, color=color, alpha=0.9)

    # Save the figure in both PNG (600 DPI) and SVG formats
    plt.savefig(f"{plot_path}.png", dpi=600, format='png')  # Save as PNG with 600 DPI
    plt.savefig(f"{plot_path}.svg", format='svg')  # Save as SVG
    plt.close()
def select_and_plot_clean_noisy(original_folder, noisy_folder, noise_info_folder, plots_folder, num_signals=90):

    # Get all the .npy files from the original folder and sort them alphabetically
    all_files = sorted([f for f in os.listdir(original_folder) if f.endswith('.npy')])

    # Select the first num_signals files in order
    selected_files = all_files[:num_signals]

    for file_name in selected_files:
        original_path = os.path.join(original_folder, file_name)
        noisy_path = os.path.join(noisy_folder, file_name)
        noise_info_path = os.path.join(noise_info_folder, os.path.splitext(file_name)[0] + '_noise_info.npy')
        plot_path = os.path.join(plots_folder, f'{os.path.splitext(file_name)[0]}_plot.png')

        if os.path.exists(original_path) and os.path.exists(noisy_path) and os.path.exists(noise_info_path):
            # Load the signals and noise info
            original_signal = np.load(original_path)
            noisy_signal = np.load(noisy_path)
            noise_info = np.load(noise_info_path, allow_pickle=True)  # Allow pickle to load object array

            plot_signals(original_signal, noisy_signal, noise_info, plot_path)
            print(f'Plotted {file_name} to {plot_path}')
        else:
            print(f'Missing files for {file_name}.')


# create the plots and ensure the folders are empty
test_plots_folder = r'C:\Users\marci\Proj_Tese\ptb_xl_360\plots\x_test_plots'
train_plots_folder = r'C:\Users\marci\Proj_Tese\ptb_xl_360\plots\x_train_plots'
val_plots_folder = r'C:\Users\marci\Proj_Tese\ptb_xl_360\plots\x_val_plots'
# List of all directories to ensure they are empty
all_folders = [test_plots_folder, train_plots_folder, val_plots_folder]
ensure_folders_empty(all_folders)

test_folder = r'C:\Users\marci\Proj_Tese\ptb_xl_360\test_selected_leads'
test_noisy_folder = r'C:\Users\marci\Proj_Tese\ptb_xl_360\x_test'
test_noise_info_folder = r'C:\Users\marci\Proj_Tese\ptb_xl_360\noise_info\x_test_noise_info'
print('starting plotting clean vs added noise for test')
select_and_plot_clean_noisy(test_folder, test_noisy_folder, test_noise_info_folder, test_plots_folder, num_signals=30)

train_folder = r'C:\Users\marci\Proj_Tese\ptb_xl_360\train_selected_leads'
train_noisy_folder = r'C:\Users\marci\Proj_Tese\ptb_xl_360\x_train'
train_noise_info_folder = r'C:\Users\marci\Proj_Tese\ptb_xl_360\noise_info\x_train_noise_info'
print('starting plotting clean vs added noise for train')
select_and_plot_clean_noisy(train_folder, train_noisy_folder, train_noise_info_folder, train_plots_folder, num_signals=30)


val_folder = r'C:\Users\marci\Proj_Tese\ptb_xl_360\val_selected_leads'
val_noisy_folder = r'C:\Users\marci\Proj_Tese\ptb_xl_360\x_val'
val_noise_info_folder = r'C:\Users\marci\Proj_Tese\ptb_xl_360\noise_info\x_val_noise_info'
print('starting plotting clean vs added noise for val')
select_and_plot_clean_noisy(val_folder, val_noisy_folder, val_noise_info_folder, val_plots_folder, num_signals=30)


##### Plot Output and Original Signal ################################################################################################
# def plot_signals_and_output(signal_clean, noisy_signal, output_array, plot_path):
#     # Define colors for each combination
#     colors = {
#         (1, 0, 0): '#ec5e42ce',  # Semi-transparent red (MA)
#         (0, 1, 0): '#fcd51599',  # Yellowish with transparency (EM)
#         (0, 0, 1): '#8eb0ddff',  # Opaque blue (BW)
#         (1, 1, 0): '#f5b980ff',  # Opaque orange (MA + EM)
#         (1, 0, 1): '#cf97e0ff',  # Opaque purple (MA + BW)
#         (0, 1, 1): '#b2c993ff',  # Opaque greenish (EM + BW)
#         (1, 1, 1): '#c5c5c5cc'  # Semi-transparent grey (MA + EM + BW)
#     }
#
#     labels = {
#         (1, 0, 0): 'MA',
#         (0, 1, 0): 'EM',
#         (0, 0, 1): 'BW',
#         (1, 1, 0): 'MA + EM',
#         (1, 0, 1): 'MA + BW',
#         (0, 1, 1): 'EM + BW',
#         (1, 1, 1): 'MA + EM + BW'
#     }
#
#     # Create subplots
#     fig, axs = plt.subplots(2, 1, figsize=(12, 12))
#
#     # Plot clean signal with highlighter effect for noise information
#     axs[0].plot(signal_clean, label='Clean Signal', color='dimgrey', linestyle='-')
#     unique_keys_clean = set()
#     for i in range(len(signal_clean)):
#         key = tuple(output_array[i])
#         if key in colors:
#             if key not in unique_keys_clean:
#                 axs[0].axvspan(i-0.5, i+0.5, color=colors[key], alpha=0.3, label=labels[key])
#                 unique_keys_clean.add(key)
#             else:
#                 axs[0].axvspan(i-0.5, i+0.5, color=colors[key], alpha=0.3 )
#     axs[0].set_xlabel('Sample Index')
#     axs[0].set_ylabel('Signal Value')
#     axs[0].set_title('Clean Signal with Noise Information')
#     handles_clean, labels_clean = axs[0].get_legend_handles_labels()
#     unique_handles_labels_clean = dict(zip(labels_clean, handles_clean)).items()  # Ensures unique labels
#     axs[0].legend([handle for label, handle in unique_handles_labels_clean], [label for label, handle in unique_handles_labels_clean])
#
#
#     # Plot noisy signal with noise information
#     axs[1].plot(noisy_signal, label='Signal Inspected', color='dimgrey', linestyle='-')
#     unique_keys_noisy = set()
#     for i in range(len(noisy_signal)):
#         key = tuple(output_array[i])
#         if key in colors:
#             axs[1].scatter(i, noisy_signal[i], color=colors[key], label=labels[key] if key not in unique_keys_noisy else "", marker='o')
#             unique_keys_noisy.add(key)
#     axs[1].set_xlabel('Sample Index')
#     axs[1].set_ylabel('Signal Value')
#     axs[1].set_title('Noisy Signal with Noise Information')
#     handles_noisy, labels_noisy = axs[1].get_legend_handles_labels()
#     unique_handles_labels_noisy = dict(zip(labels_noisy, handles_noisy)).items()  # Ensures unique labels
#     axs[1].legend([handle for label, handle in unique_handles_labels_noisy], [label for label, handle in unique_handles_labels_noisy], loc='best')
#
#     # Adjust layout and save plot
#
#     plt.savefig(plot_path)
#     plt.close()
def plot_signals_and_output(signal_clean, noisy_signal, output_array, plot_path, fs=360):
    # Define colors for each combination

    # Set font style
    plt.rcParams['font.family'] = 'Palatino Linotype'

    colors = {
        (1, 0, 0): '#ec5e42ce',  # Semi-transparent red (MA)
        (0, 1, 0): '#fcd51599',  # Yellowish with transparency (EM)
        (0, 0, 1): '#8eb0ddff',  # Opaque blue (BW)
        (1, 1, 0): '#f5b980ff',  # Opaque orange (MA + EM)
        (1, 0, 1): '#cf97e0ff',  # Opaque purple (MA + BW)
        (0, 1, 1): '#b2c993ff',  # Opaque greenish (EM + BW)
        (1, 1, 1): '#c5c5c5cc'  # Semi-transparent grey (MA + EM + BW)
    }

    labels = {
        (1, 0, 0): 'MA',
        (0, 1, 0): 'EM',
        (0, 0, 1): 'BW',
        (1, 1, 0): 'MA + EM',
        (1, 0, 1): 'MA + BW',
        (0, 1, 1): 'EM + BW',
        (1, 1, 1): 'MA + EM + BW'
    }

    # Create time array based on the sampling frequency (Fs)
    time_clean = [i / fs for i in range(len(signal_clean))]
    time_noisy = [i / fs for i in range(len(noisy_signal))]

    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))

    # Plot full clean signal first
    axs[0].plot(time_clean, signal_clean, label='Clean Signal', color='dimgrey', linestyle='-')

    # Add axvspan for active states in the clean signal plot
    for i in range(1, len(signal_clean)):
        key = tuple(output_array[i])
        if key in colors:
            # Highlight the region with axvspan
            axs[0].axvspan(time_clean[i - 1], time_clean[i], color=colors[key], alpha=0.9)

    axs[0].set_xlabel('Time (seconds)', fontsize=26)
    axs[0].set_ylabel('Signal Value', fontsize=26)
    axs[0].set_xlim(0, 10)

    # Set y-axis ticks to 0, 0.5, and 1
    axs[0].set_yticks([0, 0.5, 1])

    # Set font size for axis tick labels
    axs[0].tick_params(axis='both', which='major', labelsize=18)

    handles_clean, labels_clean = axs[0].get_legend_handles_labels()
    axs[0].legend(handles_clean, labels_clean, fontsize=32)

    # Plot full noisy signal in grey first
    axs[1].plot(time_noisy, noisy_signal, label='Noisy Signal', color='black', alpha=0.5, linestyle='-')

    # Color segments of the noisy signal where active states are present
    for i in range(1, len(noisy_signal)):
        key = tuple(output_array[i])
        if key in colors:
            # Plot a segment of the signal with the color corresponding to the current state
            axs[1].plot(time_noisy[i - 1:i + 1], noisy_signal[i - 1:i + 1], color=colors[key])

    axs[1].set_xlabel('Time (seconds)', fontsize=26)
    axs[1].set_ylabel('Signal Value', fontsize=26)
    axs[1].set_xlim(0, 10)

    # Set y-axis ticks to 0, 0.5, and 1
    axs[1].set_yticks([0, 0.5, 1])

    # Set font size for axis tick labels
    axs[1].tick_params(axis='both', which='major', labelsize=18)

    handles_noisy, labels_noisy = axs[1].get_legend_handles_labels()
    axs[1].legend(handles_noisy, labels_noisy, loc='best', fontsize=32)

    # Save the figure in both PNG (600 DPI) and SVG formats
    plt.savefig(f"{plot_path}.png", dpi=600, format='png')  # Save as PNG with 600 DPI
    plt.savefig(f"{plot_path}.svg", format='svg')  # Save as SVG
    plt.close()

def select_and_plot_output(clean_folder, signal_folder, output_folder, plots_folder, num_signals=30):
    # Get all the files from the signal folder and sort them in order
    all_files = sorted([f for f in os.listdir(signal_folder) if f.endswith('.npy')])

    # Select the first num_signals files in order
    selected_files = all_files[:num_signals]

    for file_name in selected_files:
        # Paths for the clean signal, noisy signal, output signal, and plot
        signals_clean_path = os.path.join(clean_folder, file_name)
        signals_noisy_path = os.path.join(signal_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        plot_path = os.path.join(plots_folder, f'{os.path.splitext(file_name)[0]}_plot.png')

        if os.path.exists(signals_clean_path) and os.path.exists(signals_noisy_path) and os.path.exists(output_path):
            # Load the signals and noise info
            clean_signal = np.load(signals_clean_path)
            noisy_signal = np.load(signals_noisy_path)
            output_array = np.load(output_path, allow_pickle=True)

            # Call plot function with the correct plot path
            plot_signals_and_output(clean_signal, noisy_signal, output_array, plot_path)
            print(f'Plotted {file_name} to {plot_path}')
        else:
            print(f'Missing files for {file_name}.')




# Usage example
# y_train = r'C:\Users\marci\Proj_Tese\ptb_xl_360\y_train' # output arrays
# x_train = r'C:\Users\marci\Proj_Tese\ptb_xl_360\x_train' # noisy signals
# x_train_clean = r'C:\Users\marci\Proj_Tese\ptb_xl_360\train_selected_leads' # clean signals
# y_train_plots = r'C:\Users\marci\Proj_Tese\ptb_xl_360\plots\y_train_plots'
# ensure_folders_empty([y_train_plots])
# print('starting plotting output for train')
# select_and_plot_output(x_train_clean, x_train, y_train, y_train_plots)
#
# y_test = r'C:\Users\marci\Proj_Tese\ptb_xl_360\y_test' # output arrays
# x_test = r'C:\Users\marci\Proj_Tese\ptb_xl_360\x_test' # noisy signals
# x_test_clean = r'C:\Users\marci\Proj_Tese\ptb_xl_360\test_selected_leads' # clean signals
# y_test_plots = r'C:\Users\marci\Proj_Tese\ptb_xl_360\plots\y_test_plots'
# ensure_folders_empty([y_test_plots])
# print('starting plotting output for test')
# select_and_plot_output(x_test_clean, x_test, y_test, y_test_plots)
#
#
# y_val = r'C:\Users\marci\Proj_Tese\ptb_xl_360\y_val' # output arrays
# x_val = r'C:\Users\marci\Proj_Tese\ptb_xl_360\x_val' # noisy signals
# x_val_clean = r'C:\Users\marci\Proj_Tese\ptb_xl_360\val_selected_leads' # clean signals
# y_val_plots = r'C:\Users\marci\Proj_Tese\ptb_xl_360\plots\y_val_plots'
# ensure_folders_empty([y_val_plots])
# print('starting plotting output for val')
# select_and_plot_output(x_val_clean, x_val, y_val, y_val_plots)


### Plot only one noisy file without the original signal ################################################################################################

# def plot_noisy_only(transformed_signal, noise_info):
#     # Create a figure and axis
#     fig, ax = plt.subplots(figsize=(12, 8))
#
#     # Plot the transformed signal
#     ax.plot(transformed_signal, label='Transformed Signal', alpha=0.7)
#     ax.set_title('Transformed Signal')
#     ax.set_xlabel('Sample Index')
#     ax.set_ylabel('Amplitude')
#     ax.grid(True)
#
#     # Overlay noise intervals with vertical lines and shaded regions
#     for interval in noise_info:
#         start_idx = interval[0]
#         end_idx = interval[1]
#         ax.axvspan(start_idx, end_idx, color='yellow', alpha=0.3)  # Shaded region
#         # ax.axvline(x=start_idx, color='green', linestyle='--', alpha=0.7)  # Start line
#         # ax.axvline(x=end_idx, color='red',  linestyle='--', alpha=0.7)  # End line
#
#     plt.tight_layout()
#     plt.show()

