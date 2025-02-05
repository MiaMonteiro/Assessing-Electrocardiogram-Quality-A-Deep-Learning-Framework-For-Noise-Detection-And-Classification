from tqdm import tqdm
import numpy as np
import torch
from ONE_HOT_GRU_dropout import ONEHOTgru, ECGDataset
from torch.utils.data import DataLoader
import os
from sklearn.metrics import roc_curve, auc

def calculate_roc_for_multilabel_onehot(y_true, y_scores):
    num_classes = y_true.shape[1]
    roc_info = {}
    auc_scores = {}

    for class_idx in range(num_classes):
        class_true = y_true[:, class_idx]
        class_scores = y_scores[:, class_idx]

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(class_true, class_scores)

        # Calculate AUC
        auc_score = auc(fpr, tpr)

        # Store results for each class
        roc_info[class_idx] = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
        auc_scores[class_idx] = auc_score

    return roc_info, auc_scores


def optimize_thresholds_gmean(model, dataloader, device=None):

    model.eval()
    save_probs = []
    save_y = []

    with torch.no_grad():
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            logits_ = model(X)
            probabilities = torch.sigmoid(logits_).cpu().numpy()  # Model's predicted probabilities

            save_probs.append(probabilities)
            save_y.append(Y.cpu().numpy())

    save_probs = np.concatenate(save_probs, axis=0)
    save_y = np.concatenate(save_y, axis=0)

    roc_info, _ = calculate_roc_for_multilabel_onehot(save_y, save_probs)

    # Find optimal thresholds using gmean
    optimal_thresholds = []
    gmean_scores = []

    for class_idx, roc_data in roc_info.items():
        fpr = roc_data['fpr']
        tpr = roc_data['tpr']
        thresholds = roc_data['thresholds']
        gmean = np.sqrt(tpr * (1 - fpr))

        # Find the threshold that maximizes gmean
        optimal_index = np.argmax(gmean)
        optimal_threshold = thresholds[optimal_index]
        optimal_gmean = gmean[optimal_index]

        # Store the optimal threshold and corresponding gmean score for the class
        optimal_thresholds.append(optimal_threshold)
        gmean_scores.append(optimal_gmean)

    return optimal_thresholds, gmean_scores




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


def test_model(model, device, test_loader, thresholds):

    model.eval()

    all_preds = []
    all_labels = []

    # Inference with progress bar
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Processing test data", unit="batch"):
            inputs, labels = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            preds = np.zeros_like(probabilities, dtype=int)
            for i in range(probabilities.shape[1]):  # Assuming shape (batch_size, num_classes)
                preds[:, i] = (probabilities[:, i] > thresholds[i]).astype(int)

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

    model_path = r'C:\Users\marci\paper_proj_dataset\paper\ONEHOT_GRU_layers3_hiddensize_128_states3_patience40_biderectionalTrue_drop0.3_date2810_0950'
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
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Main directory for the dataset
    main_dir = r'C:\Users\marci\paper_proj_dataset\ptb_xl_final'

    val_dataset = ECGDataset(main_dir=main_dir, subset='val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)

    optimal_thresholds, gmean_scores = optimize_thresholds_gmean(model, val_loader, device=device)
    print("Optimal thresholds for each class:", optimal_thresholds)

    test_dataset = ECGDataset(main_dir=main_dir, subset='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False, prefetch_factor=2)
    print(f"Number of test samples: {len(test_dataset)}")

    cm1 = test_model(model, device, test_loader, optimal_thresholds)


    # Calculate the metrics for the individual classes
    metrics = calculate_metrics(cm1)

    #print metrics:
    for class_name, class_metrics in metrics.items():
        print(f"Metrics for {class_name}:")
        for metric_name, value in class_metrics.items():
            print(f"  {metric_name}: {value:.5f}")


if __name__ == "__main__":
    main()
