from tqdm import tqdm
import numpy as np
import torch
from ONEHOT.ONE_HOT_GRU_dropout import ONEHOTgru, ECGDataset
from torch.utils.data import DataLoader
import os


def calculate_metrics(confusion_matrix):
    tn = confusion_matrix[0, 0]  # True Negatives
    fp = confusion_matrix[0, 1]  # False Positives
    fn = confusion_matrix[1, 0]  # False Negatives
    tp = confusion_matrix[1, 1]  # True Positives

    # Accuracy: (TP + TN) / (TP + TN + FP + FN)
    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100  # multiplied by 100 to get percentage

    # Precision: TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Sensitivity/Recall: TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
    }

    return metrics


def one_hot_to_binary(one_hot_array):

    # Convert one-hot encoded vectors into binary 0 or 1
    return (np.sum(one_hot_array, axis=-1) > 0).astype(int)


def cm_binary(confusion_matrix, y_true, y_pred):

    num_samples, num_timesteps = y_true.shape

    for i in range(num_samples):  # Iterate over each sample
        for j in range(num_timesteps):  # Iterate over each time step
            if y_true[i, j] == 0 and y_pred[i, j] == 0:
                confusion_matrix[0, 0] += 1  # True Negative
            elif y_true[i, j] == 0 and y_pred[i, j] == 1:
                confusion_matrix[0, 1] += 1  # False Positive
            elif y_true[i, j] == 1 and y_pred[i, j] == 0:
                confusion_matrix[1, 0] += 1  # False Negative
            elif y_true[i, j] == 1 and y_pred[i, j] == 1:
                confusion_matrix[1, 1] += 1  # True Positive

    return confusion_matrix


def test_model(model, device, test_loader):

    model.eval()

    all_preds = []
    all_labels = []

    # Inference with progress bar
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Processing test data", unit="batch"):
            inputs, labels = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()

            # Convert one-hot encoded labels and predictions to binary
            binary_labels = one_hot_to_binary(labels.cpu().numpy())
            binary_preds = one_hot_to_binary(preds)

            all_preds.append(binary_preds)
            all_labels.append(binary_labels)

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Initialize confusion matrix for binary classification (2x2 matrix)
    cm1 = np.zeros((2, 2), dtype=int)
    cm1 = cm_binary(cm1, all_labels, all_preds)

    return cm1


def main():
    # GPU configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model_path = r'C:\Users\marci\Proj_Tese\ptb_xl_360\model_predictions\ONEHOT_GRU_layers3_hiddensize_64_states3_patience40_biderectionalTrue_drop0.3_date0409_1751'
    model_file = 'best_model.pth'

    # Load the checkpoint
    checkpoint = torch.load(os.path.join(model_path, model_file))

    # Extract the hyperparameters
    batch_size = checkpoint['hyperparameters']['batch_size']
    hidden_size = checkpoint['hyperparameters']['hidden_size']
    num_layers = checkpoint['hyperparameters']['num_layers']
    num_states = checkpoint['hyperparameters']['num_states']
    bidirectional = checkpoint['hyperparameters']['bidirectional']

    # Initialize the model
    model = ONEHOTgru(input_size=1, hidden_size=hidden_size, num_layers=num_layers, num_states=num_states,
                      bidirectional=bidirectional).to(device)

    # Load the model state_dict from the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Main directory for the dataset
    main_dir = r'C:\Users\marci\Proj_Tese\ptb_xl_360'

    test_dataset = ECGDataset(main_dir=main_dir, subset='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False,
                             prefetch_factor=2)
    print(f"Number of test samples: {len(test_dataset)}")

    print("Testing the model and calculating the cm...")
    cm1 = test_model(model, device, test_loader)

    print("Calculating metrics for binary classification...")
    # Calculate the metrics for binary classification
    metrics = calculate_metrics(cm1)

    # Print metrics:
    print("Metrics for binary classification:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.5f}")


if __name__ == "__main__":
    main()
