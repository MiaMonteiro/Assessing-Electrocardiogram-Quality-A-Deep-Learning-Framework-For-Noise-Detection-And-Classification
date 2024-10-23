import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def plot_loss_curves(train_losses, val_losses, output_dir):
    """
    Plot and save training and validation loss curves.

    :param train_losses: List of training losses over epochs
    :param val_losses: List of validation losses over epochs
    :param output_dir: Directory to save the plot
    """
    # Set the font and color map
    plt.rcParams['font.family'] = 'Palatino Linotype'

    # Select specific colors from the colormap
    cmap = plt.cm.tab20
    color_train = cmap(0)  # First color in the colormap
    color_val = cmap(2)  # Second color in the colormap (you can choose any index you prefer)

    # Set the figure size
    plt.figure(figsize=(10, 6))

    # Plot the training and validation losses
    plt.plot(train_losses, label='Training Loss', color=color_train)
    plt.plot(val_losses, label='Validation Loss', color=color_val)

    # Increase font size for labels, title, and legend
    plt.xlabel('Epoch', fontsize=24)
    plt.ylabel('Loss', fontsize=24)
    # plt.title('Loss Curves', fontsize=16)
    plt.legend(fontsize=20)

    # Save the figure in high-resolution PNG and SVG formats
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=600)  # High-resolution PNG
    plt.savefig(os.path.join(output_dir, 'loss_curves.svg'), format='svg')  # Scalable SVG format
    plt.close()


def plot_cm4(cm, output_dir):
    class_names = ['MA', 'EM', 'BW', 'None']  # Adjust this if class names change

    plt.rcParams['font.family'] = 'Palatino Linotype'
    fig, ax = plt.subplots(figsize=(15, 10))
    base_color = "#313638"  # gray
    # Generate a custom colormap from the base color to white or lighter blue
    cmap = LinearSegmentedColormap.from_list("custom_bw", ["#ffffff", base_color])
    # Adjust the figure size as needed
    cax = ax.matshow(cm, cmap=cmap)
    fig.colorbar(cax)

    # Set labels, title, and ticks
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    # ax.set_title('Confusion Matrix')

    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    # Calculate the percentage for each cell
    cm_sum = cm.sum(axis=1)[:, np.newaxis]  # Sum of each row
    cm_percentage = cm / cm_sum.astype(float) * 100  # Calculate percentages

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percentage_text = f"{cm_percentage[i, j]:.2f}%"  # Format the percentage
            count_text = f"\n{cm[i, j]}"  # Add the count on a new line
            ax.text(j, i, percentage_text + count_text,
                    ha="center", va="center",
                    fontsize=12, fontweight='bold' if cm_percentage[i, j] > 0 else 'normal',
                    color="white" if cm[i, j] > thresh else "black",
                    fontstyle='italic' if cm[i, j] > 0 else 'normal')

    # Save the figure
    plt.savefig(os.path.join(output_dir, 'cm4.png'), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'cm4.svg'), format='svg', bbox_inches='tight')
    # plt.show()


def plot_cm1(cm1, output_dir):
    class_names = ['MA', 'EM', 'BW']
    # Set the theme and font
    plt.rcParams['font.family'] = 'Palatino Linotype'

    # Create subplots for each class
    fig, axes = plt.subplots(1, len(class_names), figsize=(15, 5))

    cmap_BW = LinearSegmentedColormap.from_list("custom_bw", ["#ffffff", '#8eb0ddff'])
    cmap_MA = LinearSegmentedColormap.from_list("custom_ma", ["#ffffff", '#ec5e42ce'])
    cmap_EM = LinearSegmentedColormap.from_list("custom_em", ["#ffffff", '#fcd515ff'])
    colormaps = [cmap_MA, cmap_EM, cmap_BW]

    # Loop through each subplot and apply the corresponding colormap
    for i, ax in enumerate(axes):
        # Extract the confusion matrix for the current class
        cm = cm1[:, :, i]

        # Select the colormap based on the current index
        cmap = colormaps[i % len(colormaps)]

        # Plot the confusion matrix using the selected colormap
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)

        # Set labels, title, and ticks
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=['Not Present', 'Present'],
               yticklabels=['Not Present', 'Present'],
               title=class_names[i])

        # Loop over data dimensions and create text annotations with larger percentage and smaller count
        total = np.sum(cm)
        for j in range(cm.shape[0]):
            for k in range(cm.shape[1]):
                count = cm[j, k]
                percentage = count / total * 100 if total > 0 else 0

                # Add the percentage with larger font size
                ax.text(k, j, f'{percentage:.2f}%',
                        ha="center", va="center",
                        fontsize=14, fontweight='bold',  # Larger, bold font for percentage
                        color="white" if count > cm.max() / 2. else "black")

                # Add the count with smaller font size below the percentage
                ax.text(k, j + 0.3, f'({count})',
                        ha="center", va="center",
                        fontsize=10, fontstyle='italic',  # Smaller, italic font for count
                        color="white" if count > cm.max() / 2. else "black")

    # Adjust layout
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cm1.png'), dpi=600, bbox_inches='tight')
    # SAVE AS SVG
    plt.savefig(os.path.join(output_dir, 'cm1.svg'), format='svg', bbox_inches='tight')


def plot_cm1_8C(cm, output_dir):
    index_to_combination = {
        0: "No noise",
        1: "BW",
        2: "EM",
        3: "EM + BW",
        4: "MA",
        5: "MA + BW",
        6: "MA + EM",
        7: "MA + EM + BW"
    }

    plt.figure(figsize=(20, 20))
    for i in range(8):
        plt.subplot(3, 3, i + 1)
        plt.imshow(cm[:, :, i], cmap="Blues", aspect='auto')

        # Annotate the confusion matrix
        for j in range(2):
            for k in range(2):
                plt.text(k, j, str(cm[j, k, i]), ha='center', va='center', color='black', fontsize=16)

        # Set the labels
        plt.xticks([0, 1], ["Other", index_to_combination[i]])
        plt.yticks([0, 1], ["Other", index_to_combination[i]])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix for {index_to_combination[i]}")

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

def plot_cm8_all(cm,output_dir):
    # Labels for the classes
    labels = [
        "No noise",
        "BW",
        "EM",
        "EM + BW",
        "MA",
        "MA + BW",
        "MA + EM",
        "MA + EM + BW"
    ]

    # Plotting the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)

    for i in range(8):
        for j in range(8):
            ax.text(j, i, str(cm[i, j]), va='center', ha='center')

    ax.set_xticks(np.arange(8))
    ax.set_yticks(np.arange(8))

    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
