import torch
import numpy as np
from tqdm import tqdm  # Import tqdm for the progress bar
from ONE_HOT_GRU_dropout import ONEHOTgru, ECGDataset
from torch.utils.data import DataLoader
import os
from eval_metrics import cm_1class, calculate_metrics, cm_4classes
from plot_metrics_model import plot_cm4, plot_cm1
from sklearn.metrics import roc_curve, auc
import numpy as np
from eval_metrics import calculate_metrics



# Threshold optimization function
def optimize_thresholds(model, device, val_loader):
    model.eval()
    class_thresholds = [0.5, 0.5, 0.5]  # Initialize with default thresholds

    all_val_preds = []
    all_val_labels = []

    # Inference on validation set
    with torch.no_grad():
        for data in tqdm(val_loader, desc="Processing validation data", unit="batch"):
            inputs, labels = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()  # Convert to probabilities

            all_val_preds.append(probs)
            all_val_labels.append(labels.cpu().numpy())

    # Concatenate all validation predictions and labels
    all_val_preds = np.concatenate(all_val_preds, axis=0)
    all_val_labels = np.concatenate(all_val_labels, axis=0)

    # Find optimal threshold for each class
    for i in range(3):  # Assuming 3 classes
        best_threshold = 0.5
        best_metric = 0

        for threshold in np.arange(0, 1.01, 0.05):  # Adjust step size as needed
            preds = (all_val_preds[:, i] > threshold).astype(int)
            # Calculate metric for this class and threshold (e.g., F1 score)
            metric = calculate_metrics(all_val_labels[:, i], preds)  # or another metric

            if metric > best_metric:
                best_metric = metric
                best_threshold = threshold

        class_thresholds[i] = best_threshold

    return class_thresholds


# Test model function with optimized thresholds for multi-label output
def test_model(model, device, test_loader, idx_samples_to_save, class_thresholds):
    model.eval()

    all_preds = []
    all_labels = []
    all_inputs = []

    # Inference with progress bar
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Processing test data", unit="batch"):
            inputs, labels = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()  # Get probabilities

            # Apply optimized thresholds independently for each class
            preds = np.zeros(probs.shape, dtype=int)
            for i in range(3):  # Assuming 3 classes
                preds[:, i] = (probs[:, i] > class_thresholds[i]).astype(int)

            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
            all_inputs.append(inputs.cpu().numpy())

    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_inputs = np.concatenate(all_inputs, axis=0)

    # Selecting the corresponding samples to save
    inputs_to_save = all_inputs[idx_samples_to_save]
    labels_to_save = all_labels[idx_samples_to_save]
    preds_to_save = all_preds[idx_samples_to_save]

    return inputs_to_save, preds_to_save, labels_to_save


def main():
    # GPU configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model_path = r'C:\Users\marci\paper_proj_dataset\models\ONEHOT_GRU_layers3_hiddensize_128_states3_patience40_biderectionalTrue_drop0.3_date0409_2154'
    model_file = 'best_model.pth'

    # Load the checkpoint
    checkpoint = torch.load(os.path.join(model_path, model_file))

    # Extract hyperparameters and initialize the model
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
    main_dir = r'C:\Users\marci\paper_proj_dataset\ptb_xl_final'

    # Validation and Test data loaders
    val_dataset = ECGDataset(main_dir=main_dir, subset='val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)

    test_dataset = ECGDataset(main_dir=main_dir, subset='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)

    # Optimize thresholds using validation set
    class_thresholds = optimize_thresholds(model, device, val_loader)
    print("Optimized thresholds:", class_thresholds)

    # Select test samples to save
    all_idx_test = len(test_loader.dataset)
    np.random.seed(42)
    idx_samples_to_save = np.random.choice(all_idx_test, 100, replace=False)
    np.save(os.path.join(model_path, 'idx_samples_to_save.npy'), idx_samples_to_save)
    idx_samples_to_save = np.load(os.path.join(model_path, 'idx_samples_to_save.npy'))

    # Run the test model function with optimized thresholds
    inputs_to_save, preds_to_save, labels_to_save = test_model(model, device, test_loader, idx_samples_to_save,
                                                               class_thresholds)

    # Save the selected inputs, labels, and predictions
    np.save(os.path.join(model_path, 'inputs100.npy'), inputs_to_save)
    np.save(os.path.join(model_path, 'labels100.npy'), labels_to_save)
    np.save(os.path.join(model_path, 'preds100.npy'), preds_to_save)


if __name__ == "__main__":
    main()
