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

def test_model(model, device, test_loader, thresholds, idx_samples_to_save):

    model.eval()

    all_preds = []
    all_labels = []
    all_inputs = []

    ## add clean signals
    all_clean_original = []

    # Inference with progress bar
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Processing test data", unit="batch"):
            inputs, labels= data
            # , clean_signals
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            preds = np.zeros_like(probabilities, dtype=int)
            for i in range(probabilities.shape[1]):  # Assuming shape (batch_size, num_classes)
                preds[:, i] = (probabilities[:, i] > thresholds[i]).astype(int)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_inputs.append(inputs.cpu().numpy())
            # all_clean_original.append(clean_signals.cpu().numpy()) # coollect clean signals


    # Concatenate all predictions and labels
    all_inputs = np.concatenate(all_inputs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    # all_clean_original = np.concatenate(all_clean_original, axis=0)

    # print('all_preds shape:', all_preds.shape)
    # print('all_labels shape:', all_labels.shape)

    # Calculate the confusion matrix for the 4 classes
    # cm4_zeros = np.zeros((4, 4), dtype=int)
    # cm4 = cm_4classes(cm4_zeros, all_labels, all_preds)

    # Plot the confusion matrix for the individual classes
    cm1 = np.zeros((2, 2, 3), dtype=int)
    cm1 = cm_1class(cm1, all_labels, all_preds)

    # Select the corresponding inputs, labels, and predictions
    inputs_to_save = all_inputs[idx_samples_to_save]
    labels_to_save = all_labels[idx_samples_to_save]
    preds_to_save = all_preds[idx_samples_to_save]
    # clean_to_save = all_clean_original[idx_samples_to_save]

    del all_inputs, all_labels, all_preds, all_clean_original
    torch.cuda.empty_cache()
    return cm1, inputs_to_save, preds_to_save, labels_to_save
# cm4,, clean_to_save
def main():

    # GPU configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model_path = r'C:\Users\marci\paper_proj_dataset\models\ONEHOT_GRU_layers3_hiddensize_128_states3_patience40_biderectionalTrue_drop0.3_date0409_2154'
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

    # Load the model state_dict from the checkpoint


    # test_dataset = ECGDataset(main_dir=main_dir, subset='test')

    # Limit the amount of data being loaded (e.g., first 1000 samples)
    # subset_size = 1000
    # indices = list(range(len(test_dataset)))
    # test_subset = Subset(test_dataset, indices[:subset_size])

    test_dataset = ECGDataset(main_dir=main_dir, subset='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False, prefetch_factor=2)
    print(f"Number of test samples: {len(test_dataset)}")

    all_idx_test = len(test_loader.dataset)
    # Ensure consistent sample selection
    np.random.seed(42)
    idx_samples_to_save = np.random.choice(all_idx_test, 100, replace=False)
    #
    # #print how many are being selected
    print(len(idx_samples_to_save), "Signals have been selected")
    # print('input index:', idx_samples_to_save)
    # # save this indexing
    np.save(os.path.join(model_path, 'idx_samples_to_save.npy'), idx_samples_to_save)

    idx_samples_to_save = np.load(os.path.join(model_path, 'idx_samples_to_save.npy'))

    #     # Run the test model function
    # cm4,, inputs_to_save, preds_to_save, labels_to_save, clean_to_save
    inputs_to_save, preds_to_save, labels_to_save = test_model(model, device, test_loader, idx_samples_to_save,optimal_thresholds)
    #, idx_samples_to_save

    # print the shapes
    # print("Inputs shape", inputs_to_save.shape)
    # print("Preds shape", preds_to_save.shape)
    # print("Labels shape", labels_to_save.shape)
    # print("Clean signals shape", clean_to_save.shape)
    #
    # # Save the selected inputs, labels, and predictions
    np.save(os.path.join(model_path, 'inputs100.npy'), inputs_to_save)
    np.save(os.path.join(model_path, 'labels100.npy'), labels_to_save)
    np.save(os.path.join(model_path, 'preds100.npy'), preds_to_save)
    # np.save(os.path.join(model_path, 'clean_signals1000.npy'), clean_to_save)


    #add o ploting aqui


    # Plot the confusion matrices

    # plot_cm4(cm4, model_path)
    # plot_cm1(cm1, model_path)

    # Calculate the metrics for the individual classes
    # metrics = calculate_metrics(cm1)
    #
    # #print metrics:
    # for class_name, class_metrics in metrics.items():
    #     print(f"Metrics for {class_name}:")
    #     for metric_name, value in class_metrics.items():
    #         print(f"  {metric_name}: {value:.5f}")


if __name__ == "__main__":
    main()

    # for class_name, class_metrics in metrics.items():
    #     logging.info(f"Metrics for {class_name}:")
    #     for metric_name, value in class_metrics.items():
    #         logging.info(f"  {metric_name}: {value:.2f}")
    #
    # logging.info("Model testing process completed.")




