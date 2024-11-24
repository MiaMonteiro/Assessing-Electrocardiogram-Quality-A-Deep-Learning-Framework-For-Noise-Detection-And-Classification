import matplotlib.pyplot as plt
import numpy as np
import os


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
    axs[0].plot(time, original_signal, label='Original Signal', color='black', alpha=0.7)
    axs[0].set_ylabel('Amplitude', fontsize=26)
    axs[0].set_xlim(0, 10)  # Set x-axis limits to 0 to 10 seconds
    # axs[0].set_yticks([0, 0.5, 1])  # Set y-axis ticks
    axs[0].tick_params(axis='both', which='major', labelsize=18)  # Set tick size
    # axs[0].grid(True)

    # Plot the transformed (noisy) signal with time on x-axis
    axs[1].plot(time, transformed_signal, label='Transformed Signal', color='black', alpha=0.7)
    axs[1].set_xlabel('Time (seconds)', fontsize=26)  # Set x-axis label with fontsize 26
    axs[1].set_ylabel('Amplitude', fontsize=26)  # Set y-axis label with fontsize 26
    axs[1].set_xlim(0, 10)  # Set x-axis limits to 0 to 10 seconds
    # axs[1].set_yticks([0, 0.5, 1])  # Set y-axis ticks
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


def select_and_plot_clean_noisy(original_folder, noisy_folder, noise_info_folder, plots_folder, num_signals=150):

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
x_test_plots_folder = r'C:\Users\marci\paper_proj_dataset\plots_clean_noisy\x_test_plots'
x_train_plots_folder = r'C:\Users\marci\paper_proj_dataset\plots_clean_noisy\x_train_plots'
x_val_plots_folder = r'C:\Users\marci\paper_proj_dataset\plots_clean_noisy\x_val_plots'
all_folders = [x_test_plots_folder, x_train_plots_folder, x_val_plots_folder]
for folder in all_folders:
    os.makedirs(folder, exist_ok=True)

x_test_folder = r'C:\Users\marci\paper_proj_dataset\ptb_xl_clean_360_normalized\x_test_clean'
x_test_noisy_folder = r'C:\Users\marci\paper_proj_dataset\ptb_xl_noisy_360_normalized\x_test_noisy'
x_test_noise_info_folder = r'C:\Users\marci\paper_proj_dataset\ptb_xl_noisy_360\noise_info\x_test_noise_info'
print('starting plotting clean vs added noise for test')
select_and_plot_clean_noisy(x_test_folder, x_test_noisy_folder, x_test_noise_info_folder, x_test_plots_folder)

x_train_folder = r'C:\Users\marci\paper_proj_dataset\ptb_xl_clean_360_normalized\x_train_clean'
x_train_noisy_folder = r'C:\Users\marci\paper_proj_dataset\ptb_xl_noisy_360_normalized\x_train_noisy'
x_train_noise_info_folder = r'C:\Users\marci\paper_proj_dataset\ptb_xl_noisy_360\noise_info\x_train_noise_info'
print('starting plotting clean vs added noise for train')
select_and_plot_clean_noisy(x_train_folder, x_train_noisy_folder, x_train_noise_info_folder, x_train_plots_folder)

x_val_folder = r'C:\Users\marci\paper_proj_dataset\ptb_xl_clean_360_normalized\x_val_clean'
x_val_noisy_folder = r'C:\Users\marci\paper_proj_dataset\ptb_xl_noisy_360_normalized\x_val_noisy'
x_val_noise_info_folder = r'C:\Users\marci\paper_proj_dataset\ptb_xl_noisy_360\noise_info\x_val_noise_info'
print('starting plotting clean vs added noise for val')
select_and_plot_clean_noisy(x_val_folder, x_val_noisy_folder, x_val_noise_info_folder, x_val_plots_folder)


