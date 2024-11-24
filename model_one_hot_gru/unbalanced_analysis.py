import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm  # Import tqdm for progress bar
import matplotlib.font_manager as fm
from matplotlib.colors import to_rgba


def analyze_dataset(file_path):
    # Load the one-hot encoded data
    y_set = np.load(file_path)

    # Initialize counters for each type of noise
    noise_counts = {
        'MA': 0,  # Corresponds to [1, 0, 0]
        'EM': 0,  # Corresponds to [0, 1, 0]
        'BW': 0,  # Corresponds to [0, 0, 1]
        'None': 0  # Corresponds to [0, 0, 0]
    }

    for vec in y_set:
        if np.array_equal(vec, [0, 0, 0]):
            noise_counts['None'] += 1
        if vec[0] == 1:
            noise_counts['MA'] += 1
        if vec[1] == 1:
            noise_counts['EM'] += 1
        if vec[2] == 1:
            noise_counts['BW'] += 1

    return noise_counts


def main():
    # Define the directory for training data
    train_dir = r'C:\Users\marci\paper_proj_dataset\ptb_xl_y_classification\y_3600\y_train'

    # List all .npy files in the training directory
    file_names = [f for f in os.listdir(train_dir) if f.endswith('.npy')]

    # Initialize counters for the training dataset
    noise_counts = {'MA': 0, 'EM': 0, 'BW': 0, 'None': 0}

    # Initialize the progress bar
    for file_name in tqdm(file_names, desc="Processing files"):
        file_path = os.path.join(train_dir, file_name)
        file_noise_counts = analyze_dataset(file_path)

        for key in noise_counts:
            noise_counts[key] += file_noise_counts.get(key, 0)

    # Print out the results
    print("Noise counts for training dataset:")
    for noise_type, count in noise_counts.items():
        print(f"  {noise_type}: {count}")


if __name__ == "__main__":
    main()


#### PLOT HIST #####
150233577
# categories = ['MA', 'EM', 'BW', 'None']
# percentages = [16.0, 16.0, 27.4, 40.7]
# actual_numbers = [31531853, 31127919, 53146197, 79333726]
# colors = ['lightcoral', 'palegreen', 'paleturquoise', 'lightgray']
#
# # Apply alpha to colors
# colors_with_alpha = [to_rgba(c, alpha=0.5) for c in colors]
#
# # Font settings
# font_properties = fm.FontProperties(fname=fm.findfont(fm.FontProperties(family='Palatino Linotype')), size=32)
#
# # Create figure and axis with a smaller size
# fig, ax = plt.subplots(figsize=(14, 10))  # Adjust the size here
#
# # Define the bar width and spacing
# bar_width = 0.6 # Make bars skinnier
# bar_spacing = 0.4  # Adjust this to make bars closer
#
# # Create the histogram with specified colors, transparency, and width
# bars = ax.bar(categories, percentages, color=colors_with_alpha, width=bar_width)
#
# # Add percentage labels inside the bars
# for bar, percentage in zip(bars, percentages):
#     height = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width() / 2, height / 2,
#             f'{percentage}%',  # Add percentage label inside the bar
#             ha='center', va='center', color='black', fontproperties=font_properties)
#
# # # Add actual number labels on top of the bars
# # for bar, number in zip(bars, actual_numbers):
# #     height = bar.get_height()
# #     ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
# #             f'{number:,}',  # Format number with commas
# #             ha='center', va='bottom', color='black', fontproperties=font_properties)
#
# # Set labels and title with the specified font
# ax.set_xlabel('Noise Types', fontproperties=font_properties)
# ax.set_ylabel('Percentage (%)', fontproperties=font_properties)
# # ax.set_title('Histogram with Percentages Inside Bars and Numbers on Top', fontproperties=font_properties)
#
# ax = plt.gca()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
#
# # Apply font properties to axis tick labels
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#     label.set_fontproperties(font_properties)
#
# # Adjust the x-axis limits to fit the bars closer
# ax.set_xlim(-bar_spacing, len(categories) - 1 + bar_spacing)  # Add spacing to the x-axis limits
#
# # Adjust the y-axis limits to add space above the bars
# ylim = ax.get_ylim()  # Get current y-axis limits
# ax.set_ylim(0, ylim[1] * 1.1)  # Increase the upper limit by 10%
# plt.savefig('class_imbalance.png', dpi=300, bbox_inches='tight')
# # Show plot
# plt.show()
# save plot


