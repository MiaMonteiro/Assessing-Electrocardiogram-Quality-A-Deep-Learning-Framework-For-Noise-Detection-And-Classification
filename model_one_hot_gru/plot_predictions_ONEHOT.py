import numpy as np
import matplotlib.pyplot as plt
import os
from classification import classify
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker


def add_label_spans(ax, labels, label_names, color_dict, alpha=0.6, label_prefix="", fs=360):
    start = 0
    current_label = labels[0]
    present_labels = set()  # Track present labels for legend
    legend_patches = []  # Store patches for the legend

    for i in range(1, len(labels)):
        if not np.array_equal(labels[i], current_label) or i == len(labels) - 1:
            end = i if not np.array_equal(labels[i], current_label) else i + 1

            # Convert sample indices to time
            start_time = start / fs
            end_time = end / fs

            active_labels = [idx for idx, value in enumerate(current_label) if value == 1]

            if len(active_labels) > 0:
                # Handle single noise type or combination
                if len(active_labels) == 1:
                    label = label_names[active_labels[0]]
                    color = color_dict[active_labels[0]]
                else:
                    # Create combination label
                    label = " + ".join([label_names[idx] for idx in active_labels])
                    if len(active_labels) == 2:
                        if (0 in active_labels) and (1 in active_labels):  # MA + EM
                            color = '#F1A156'
                        elif (0 in active_labels) and (2 in active_labels):  # MA + BW
                            color = '#B270C6'
                        elif (1 in active_labels) and (2 in active_labels):  # EM + BW
                            color = '#B2C993'
                    elif len(active_labels) == 3:  # MA + EM + BW
                        color = '#AFAEAD'
                ax.axvspan(start_time, end_time, color=color, alpha=alpha)

                if label not in present_labels:
                    present_labels.add(label)
                    # Correctly convert the color to RGBA using mcolors.to_rgba
                    rgba_color = tuple(np.array(mcolors.to_rgba(color)) * np.array([1, 1, 1, alpha]))
                    legend_patches.append(mpatches.Patch(color=rgba_color, label=f"{label_prefix} {label}"))

            start = end
            current_label = labels[i]

    return legend_patches  # Return legend patches


# Define label colors and names
# label_colors = {0: plt.get_cmap('Accent')(3), 1: plt.get_cmap('tab20b')(18), 2: plt.get_cmap('tab20c')(0)}

# Define label colors and names using hex codes
label_colors = {
    0: '#E86146',  # Example hex color for MA
    1: '#FAE166',  # Example hex color for EM
    2: '#6697D8'   # Example hex color for BW
}

label_names = ['MA', 'EM', 'BW']



plt.rcParams['font.family'] = 'Palatino Linotype'


# def plot_label_spans(inputdir, idx, alpha=0.5, save_path=None, fs=360):
#     inputs = np.load(os.path.join(inputdir, 'inputs.npy'))
#     labels = np.load(os.path.join(inputdir, 'labels.npy'))
#     preds = np.load(os.path.join(inputdir, 'preds.npy'))
#     clean_signals = np.load(os.path.join(inputdir, 'clean_signals.npy'))  # Load clean signals
#
#     # Ensure inputs have correct shape
#     assert inputs.shape[0] == labels.shape[0] == preds.shape[0] == clean_signals.shape[0] == len(idx), "Mismatch between idx and data size."
#
#     num_samples = inputs.shape[1]  # Number of samples in the signal
#     time_values = np.arange(num_samples) / fs  # Generate time values in seconds
#
#     for i in range(inputs.shape[0]):  # Loop through all signals (300 signals)
#         # Classify the signal
#         classification_result = classify(preds[i])
#
#         # Create a 2-column layout with 3 rows, using gridspec
#         fig = plt.figure(figsize=(15, 10))  # Adjusted size to better fit the text and plots
#         gs = gridspec.GridSpec(3, 2, width_ratios=[3, 1])  # 2 columns, 3 rows. Text box will go in the second column.
#
#         # Clean signal plot in the first row, first column
#         ax1 = plt.subplot(gs[0, 0])
#         ax1.plot(time_values, clean_signals[i].squeeze(), label='Clean Signal', color='black', alpha=0.5)  # Use time_values for x-axis
#         ax1.set_title(f'Clean Signal - File: {idx[i]}')
#         ax1.legend()
#         ax1.set_xlabel('Time (seconds)')
#         ax1.set_ylabel('Amplitude')
#
#         # Adjust the text box's axis position
#         ax_text = plt.subplot(gs[0, 1])  # Text box in the second column, first row
#         ax_text.axis('off')  # Turn off the axis for the text box
#
#         bbox_props = dict(boxstyle='square,pad=1', facecolor='none', edgecolor='black', alpha=0.5, linewidth=1)
#
#         # Add text aligned to the left (ha='left') with more space around it using bbox_props
#         ax_text.text(0, 0.5, classification_result['report'], fontsize=14,
#                      ha='left', va='center', bbox=bbox_props, wrap=True)
#
#         gs.update(left=0.0, right=1, top=0.95, bottom=0.05, wspace=0.1, hspace=0.3)
#
#         # Plot the true labels on noisy signal in the second row, spanning both columns
#         ax2 = plt.subplot(gs[1, :2], sharex=ax1)
#         ax2.plot(time_values, inputs[i].squeeze(), label='Noisy Signal', color='black', alpha=0.5)  # Use time_values for x-axis
#         add_label_spans(ax2, labels[i], label_names, label_colors, alpha=alpha, label_prefix="True")
#         ax2.set_title(f'True Labels on Noisy Signal - Signal {idx[i]}')
#         ax2.set_ylabel('Amplitude')
#         ax2.legend(loc='upper right')
#         ax2.set_xlabel('Time (seconds)')
#
#         # Plot the predicted labels on noisy signal in the third row, spanning both columns
#         ax3 = plt.subplot(gs[2, :2], sharex=ax1)
#         ax3.plot(time_values, inputs[i].squeeze(), label='Noisy Signal', color='black', alpha=0.5)  # Use time_values for x-axis
#         add_label_spans(ax3, preds[i], label_names, label_colors, alpha=alpha, label_prefix="Predicted")
#         ax3.set_title(f'Predicted Labels on Noisy Signal - Signal {idx[i]}')
#         ax3.set_xlabel('Time (seconds)')
#         ax3.set_ylabel('Amplitude')
#         ax3.legend(loc='upper right')
#
#         # Save the plot with the filename matching the index in idx
#         if save_path:
#             full_save_path = os.path.join(save_path, f"{idx[i]}_classified.png")
#             plt.savefig(full_save_path, dpi=300, bbox_inches='tight')  # High-quality save with adjusted layout
#             print(f"Plot saved to {full_save_path}")
#
#         plt.close(fig)

 # OPTIMZED FOR VIEWING
# def plot_label_spans(inputdir, idx, save_path=None, fs=360):
#     inputs = np.load(os.path.join(inputdir, 'inputs.npy'))
#     labels = np.load(os.path.join(inputdir, 'labels.npy'))
#     preds = np.load(os.path.join(inputdir, 'preds.npy'))
#     clean_signals = np.load(os.path.join(inputdir, 'clean_signals.npy'))  # Load clean signals
#
#     # Ensure inputs have correct shape
#     assert inputs.shape[0] == labels.shape[0] == preds.shape[0] == clean_signals.shape[0] == len(idx), "Mismatch between idx and data size."
#
#     num_samples = inputs.shape[1]  # Number of samples in the signal
#     time_values = np.arange(num_samples) / fs  # Generate time values in seconds
#
#     for i in range(inputs.shape[0]):  # Loop through all signals (300 signals)
#         # Classify the signal
#         classification_result = classify(preds[i])
#
#         # Create a 2-column layout with 3 rows, using gridspec
#         fig = plt.figure(figsize=(8, 10))  # Adjusted size to better fit the text and plots
#         gs = gridspec.GridSpec(3, 2, width_ratios=[3, 1])  # 2 columns, 3 rows. Text box will go in the second column.
#
#         # Clean signal plot in the first row, first column
#         ax1 = plt.subplot(gs[0, 0])
#         ax1.plot(time_values, clean_signals[i].squeeze(), label='Clean Signal', color='black', alpha=0.5)  # Use time_values for x-axis
#         # ax1.set_title(f'Clean Signal - File: {idx[i]}')
#         ax1.legend(fontsize=24)
#         ax1.tick_params(axis='x', labelsize=18)  # Increase x-axis tick font size
#         ax1.set_yticks([0, 0.5, 1])
#         ax1.tick_params(axis='y', labelsize=18)
#         # ax1.set_xlabel('Time (seconds)', fontsize = 24)
#         # ax1.set_ylabel('Amplitude')
#         ax1.set_xlim(0, 10)
#
#
#         # Adjust the text box's axis position
#         ax_text = plt.subplot(gs[0, 1])  # Text box in the second column, first row
#         ax_text.axis('off')  # Turn off the axis for the text box
#
#         bbox_props = dict(boxstyle='square,pad=1', facecolor='none', edgecolor='black', alpha=0.5, linewidth=1)
#
#         # Add text aligned to the left (ha='left') with more space around it using bbox_props
#         ax_text.text(0, 0.5, classification_result['report'], fontsize=14, ha='left', va='center', bbox=bbox_props, wrap=True)
#
#         gs.update(left=0.0, right=1, top=0.95, bottom=0.05, wspace=0.1, hspace=0.3)
#
#         # Plot the true labels on noisy signal in the second row, spanning both columns
#         ax2 = plt.subplot(gs[1, :2], sharex=ax1)
#         ax2.plot(time_values, inputs[i].squeeze(), label='Noisy Signal', color='black', alpha=0.5)  # Use time_values for x-axis
#
#         # Get present labels for true labels and convert to time using fs
#         true_label_patches = add_label_spans(ax2, labels[i], label_names, label_colors, label_prefix="True", fs=fs)
#         # ax2.set_title(f'True Labels on Noisy Signal - Signal {idx[i]}')
#         ax2.set_ylabel('Amplitude', fontsize = 32)
#         # ax2.tick_params(axis='x', length=0, labelbottom=False)  # Increase x-axis tick font size
#         ax2.set_xlabel('Time (seconds)', fontsize = 32)
#         ax2.tick_params(axis='x', labelsize=26)
#         ax2.set_yticks([0, 0.5, 1])
#         ax2.tick_params(axis='y', labelsize=26)
#         ax2.set_xlim(0, 10)
#
#          # Increase x-axis tick font size
#
#         # Only show legend for present labels
#         if true_label_patches:
#             ax2.legend(handles=true_label_patches, loc='upper right', fontsize=36)
#
#         ax2.set_xlabel('Time (seconds)', fontsize = 32)
#
#
#         # # Customize grid appearance (optional)
#         # ax2.grid(True, which='both', axis='both', linestyle='-', linewidth=0.5, color='gray', alpha=0.6)
#         #
#         # # Set custom tick spacing for the x-axis and y-axis
#         # ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))  # Set x-axis ticks at intervals of 1
#         # ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.1))  # Set y-axis ticks at intervals of 0.2
#         #
#         # ax2.grid(True)
#         # Plot the predicted labels on noisy signal in the third row, spanning both columns
#         ax3 = plt.subplot(gs[2, :2], sharex=ax1)
#         ax3.plot(time_values, inputs[i].squeeze(), label='Noisy Signal', color='black', alpha=0.5)  # Use time_values for x-axis
#
#         # Get present labels for predicted labels and convert to time using fs
#         pred_label_patches = add_label_spans(ax3, preds[i], label_names, label_colors, label_prefix="Predicted", fs=fs)
#         # ax3.set_title(f'Predicted Labels on Noisy Signal - Signal {idx[i]}')
#         ax3.set_xlabel('Time (seconds)', fontsize = 32)
#         ax3.tick_params(axis='x', labelsize=26)  # Increase x-axis tick font size
#         ax3.set_ylabel('Amplitude', fontsize=32)
#         ax3.set_yticks([0, 0.5, 1])
#         ax3.tick_params(axis='y', labelsize=26)
#         ax3.set_xlim(0, 10)
#
#
#         # Customize grid appearance (optional)
#         # ax3.grid(True, which='both', axis='both', linestyle='-', linewidth=0.5, color='gray', alpha=0.6)
#         #
#         # # Set custom tick spacing for the x-axis and y-axis
#         # ax3.xaxis.set_major_locator(ticker.MultipleLocator(1))  # Set x-axis ticks at intervals of 1
#         # ax3.yaxis.set_major_locator(ticker.MultipleLocator(0.1))  # Set y-axis ticks at intervals of 0.2
#         #
#         # ax3.grid(True)
#         # Only show legend for present labels
#         if pred_label_patches:
#             ax3.legend(handles=pred_label_patches, loc='upper right', fontsize=36)
#
#         # Save the plot with the filename matching the index in idx
#         if save_path:
#             # full_save_path = os.path.join(save_path, f"{idx[i]}_classified.png")
#             # plt.savefig(full_save_path, dpi=300, bbox_inches='tight')  # High-quality save with adjusted layout
#             # print(f"Plot saved to {full_save_path}")
#
#
#             full_save_path = os.path.join(save_path, f"{idx[i]}_classified.svg")  # Change extension to .svg
#             plt.savefig(full_save_path, format='svg', bbox_inches='tight')  # Save as SVG
#             print(f"Plot saved to {full_save_path}")
#
#         plt.close(fig)

# Clean no text box


def plot_label_spans(inputdir, idx, save_path=None, fs=360):
    inputs = np.load(os.path.join(inputdir, 'inputs.npy'))
    labels = np.load(os.path.join(inputdir, 'labels.npy'))
    preds = np.load(os.path.join(inputdir, 'preds.npy'))
    clean_signals = np.load(os.path.join(inputdir, 'clean_signals.npy'))  # Load clean signals

    # Ensure inputs have correct shape
    assert inputs.shape[0] == labels.shape[0] == preds.shape[0] == clean_signals.shape[0] == len(idx), "Mismatch between idx and data size."

    num_samples = inputs.shape[1]  # Number of samples in the signal
    time_values = np.arange(num_samples) / fs  # Generate time values in seconds

    for i in range(inputs.shape[0]):  # Loop through all signals (300 signals)
        # Classify the signal
        classification_result = classify(preds[i])

        # Create a layout with 3 rows and 1 column (since we removed the text box)
        fig = plt.figure(figsize=(8, 10))  # Adjusted size to better fit the plots
        gs = gridspec.GridSpec(3, 1)  # 1 column, 3 rows. No second column for the text box.

        # Clean signal plot in the first row
        ax1 = plt.subplot(gs[0])
        ax1.plot(time_values, clean_signals[i].squeeze(), label='Clean Signal', color='black', alpha=0.5)  # Use time_values for x-axis
        ax1.legend(fontsize=24)
        ax1.tick_params(axis='x', labelsize=18)  # Increase x-axis tick font size
        ax1.set_yticks([0, 0.5, 1])
        ax1.tick_params(axis='y', labelsize=18)
        ax1.set_xlim(0, 10)

        # Plot the true labels on noisy signal in the second row
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.plot(time_values, inputs[i].squeeze(), label='Noisy Signal', color='black', alpha=0.5)  # Use time_values for x-axis

        # Get present labels for true labels and convert to time using fs
        true_label_patches = add_label_spans(ax2, labels[i], label_names, label_colors, label_prefix="True", fs=fs)
        ax2.set_ylabel('Amplitude', fontsize=32)
        ax2.tick_params(axis='x', labelsize=26)
        ax2.set_yticks([0, 0.5, 1])
        ax2.tick_params(axis='y', labelsize=26)
        ax2.set_xlim(0, 10)

        # Only show legend for present labels
        if true_label_patches:
            ax2.legend(handles=true_label_patches, loc='upper right', fontsize=36)

        ax2.set_xlabel('Time (seconds)', fontsize=32)

        # Plot the predicted labels on noisy signal in the third row
        ax3 = plt.subplot(gs[2], sharex=ax1)
        ax3.plot(time_values, inputs[i].squeeze(), label='Noisy Signal', color='black', alpha=0.5)  # Use time_values for x-axis

        # Get present labels for predicted labels and convert to time using fs
        pred_label_patches = add_label_spans(ax3, preds[i], label_names, label_colors, label_prefix="Predicted", fs=fs)
        ax3.set_xlabel('Time (seconds)', fontsize=32)
        ax3.tick_params(axis='x', labelsize=26)  # Increase x-axis tick font size
        ax3.set_ylabel('Amplitude', fontsize=32)
        ax3.set_yticks([0, 0.5, 1])
        ax3.tick_params(axis='y', labelsize=26)
        ax3.set_xlim(0, 10)

        # Only show legend for present labels
        if pred_label_patches:
            ax3.legend(handles=pred_label_patches, loc='upper right', fontsize=36)

        # Save the plot with the filename matching the index in idx
        if save_path:

            full_save_path = os.path.join(save_path, f"{idx[i]}_classified.svg")  # Change extension to .svg
            plt.savefig(full_save_path, format='svg', bbox_inches='tight')  # Save as SVG
            print(f"Plot saved to {full_save_path}")

            # full_save_path = os.path.join(save_path, f"{idx[i]}_classified.png")
            # plt.savefig(full_save_path, dpi=300, bbox_inches='tight')  # High-quality save with adjusted layout
            # print(f"Plot saved to {full_save_path}")
        plt.close(fig)



# Define the input and output directories
inputdir = r'C:\Users\marci\paper_proj_dataset\models\ONEHOT_GRU_layers3_hiddensize_128_states3_patience40_biderectionalTrue_drop0.3_date0409_215'
# Load the indices to plot
idx = np.load(os.path.join(inputdir, 'idx_samples_to_save.npy'))

output_dir = r'C:\Users\marci\paper_proj_dataset\models\ONEHOT_GRU_layers3_hiddensize_128_states3_patience40_biderectionalTrue_drop0.3_date0409_215\plots'

# Call the plotting function with the indices and save path
plot_label_spans(inputdir, idx=idx, save_path=output_dir)
