from scipy.signal import resample_poly
from sklearn.preprocessing import minmax_scale
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_12_leads(signals, fs_original=500, fs_target=360, target_duration=10):
    # Ensure we calculate the correct number of samples after downsampling
    plt.rcParams['font.family'] = 'Palatino Linotype'
    num_samples_original = signals.shape[1]
    expected_samples_original = int(target_duration * fs_original)

    if num_samples_original != expected_samples_original:
        raise ValueError(
            f"Expected {expected_samples_original} samples, but got {num_samples_original}. Check the input signal.")

    num_samples_target = int(target_duration * fs_target)

    # Downsample each lead from fs_original to fs_target
    downsampled_signals = np.zeros((12, num_samples_target))
    for i in range(12):
        # Ensure proper resampling without distortion
        downsampled_signals[i, :] = resample_poly(signals[i, :], fs_target, fs_original)

    # Apply Min-Max scaling to each downsampled lead
    scaled_signals = np.array([minmax_scale(lead) for lead in downsampled_signals])

    # Create the time array for 10 seconds duration
    time = np.linspace(0, target_duration, num_samples_target)  # Time array from 0 to 10 seconds

    # Debug: Print time array range
    print(f"Time array ranges from {time[0]:.2f} to {time[-1]:.2f} seconds.")

    # Create a 4x3 grid of subplots for 12 leads
    fig, axs = plt.subplots(4, 3, figsize=(18, 12), sharex=True)

    # Define common properties
    y_ticks = [0, 0.5, 1]
    legend_fontsize = 26
    tick_fontsize = 18
    axis_fontsize = 26

    # Plot each lead in its own subplot
    for i in range(12):
        row = i // 3  # Determine which row in the grid
        col = i % 3  # Determine which column in the grid

        ax = axs[row, col]
        ax.plot(time, scaled_signals[i, :], label=f'Lead {i + 1}', color='black', alpha=0.7)

        # Set y-axis ticks
        ax.set_yticks(y_ticks)

        # Set tick sizes
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        # Set axis labels only on the edges to avoid clutter
        if row == 3:
            ax.set_xlabel('Time (seconds)', fontsize=axis_fontsize)
        if col == 0:
            ax.set_ylabel('Amplitude', fontsize=axis_fontsize)

        # Add legend
        ax.legend(fontsize=legend_fontsize)

    # Adjust layout to be compact
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.2)  # Adjust space between plots

    # Save the plot in both PNG (600 DPI) and SVG formats
    plt.savefig(f'10079_test_all_leads.png', dpi=600, format='png')
    plt.savefig(f'10079_test_all_leads.svg', format='svg')
    plt.close()
    print(f"Saved 12-lead plot")


# Example usage with your signal data and appropriate fs_original, fs_target
# plot_12_leads(signals)

# Example usage:
dir_path = r'C:\Users\marci\Proj_Tese\ptb_xl_360_split\test_all_leads'
filename = '10079.npy'
signals = np.load(os.path.join(dir_path, filename))
print(signals.shape)

# Call the plotting function with downsampling
plot_12_leads(signals.T, fs_original=500, fs_target=360)

