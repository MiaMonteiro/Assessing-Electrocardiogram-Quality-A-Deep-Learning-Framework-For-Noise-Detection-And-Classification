import numpy as np

def cm_1class(confusion_matrix, y_true, y_pred):
    num_samples, num_timesteps, num_classes = y_true.shape

    for i in range(num_samples):  # Iterate over each sample
        for j in range(num_timesteps):  # Iterate over each time step
            for k in range(num_classes):  # Iterate over each class
                if y_true[i, j, k] == 0 and y_pred[i, j, k] == 0:
                    confusion_matrix[0, 0, k] += 1  # True Negative for class k
                elif y_true[i, j, k] == 0 and y_pred[i, j, k] == 1:
                    confusion_matrix[0, 1, k] += 1  # False Positive for class k
                elif y_true[i, j, k] == 1 and y_pred[i, j, k] == 0:
                    confusion_matrix[1, 0, k] += 1  # False Negative for class k
                elif y_true[i, j, k] == 1 and y_pred[i, j, k] == 1:
                    confusion_matrix[1, 1, k] += 1  # True Positive for class k

    return confusion_matrix


def calculate_metrics(confusion_matrix):
    num_classes = confusion_matrix.shape[2]
    metrics = {}

    for i in range(num_classes):
        tn = confusion_matrix[0, 0, i]  # True Negatives for class i
        fp = confusion_matrix[0, 1, i]  # False Positives for class i
        fn = confusion_matrix[1, 0, i]  # False Negatives for class i
        tp = confusion_matrix[1, 1, i]  # True Positives for class i

        # Accuracy: (TP + TN) / (TP + TN + FP + FN)
        accuracy = (tp + tn) / (tp + tn + fp + fn) * 100  # multiplied by 100 to get percentage

        # Precision: TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # Sensitivity/Recall: TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics[f'class_{i}'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
        }

    return metrics


def cm_4classes(confusion_matrix, y_true, y_pred):
    for signal_idx in range(y_pred.shape[0]): #y_pred.shape[0] = number of signals
        true_label_signal = y_true[signal_idx]
        pred_label_signal = y_pred[signal_idx]

        for sample_idx in range(true_label_signal.shape[0]):
            # Get the true label (one-hot vector) for the current sample
            true_label = true_label_signal[sample_idx]
            # Get the predicted label (one-hot vector) for the current sample
            pred_label = pred_label_signal[sample_idx]

            if np.sum(true_label) == 0:  # Check if true label is 'none'
                if np.sum(pred_label) == 0:  # Check if prediction is also 'none'
                    confusion_matrix[3, 3] += 1 # True positive for 'none'
                else:
                    # False negative for 'none', update based on active predicted class
                    if pred_label[0] == 1:
                        confusion_matrix[3, 0] += 1  # Predicted 'ma'
                    if pred_label[1] == 1:
                        confusion_matrix[3, 1] += 1  # Predicted 'em'
                    if pred_label[2] == 1:
                        confusion_matrix[3, 2] += 1  # Predicted 'bw'
            else:  # True label is not 'none'
                if true_label[0] == 1:  # Check if true label is 'ma'
                    if pred_label[0] == 1:
                        confusion_matrix[0, 0] += 1  # True positive for 'ma'
                    else:
                        # False negative for 'ma', update based on predicted class
                        if np.sum(pred_label) == 0:
                            confusion_matrix[0, 3] += 1  # Predicted 'none'
                        if pred_label[1] == 1:
                            confusion_matrix[0, 1] += 1  # Predicted 'em'
                        if pred_label[2] == 1:
                            confusion_matrix[0, 2] += 1  # Predicted 'bw'
                if true_label[1] == 1:  # Check if true label is 'em'
                    if pred_label[1] == 1:
                        confusion_matrix[1, 1] += 1  # True positive for 'em'
                    else:
                        # False negative for 'em', update based on predicted class
                        if np.sum(pred_label) == 0:
                            confusion_matrix[1, 3] += 1  # Predicted 'none'
                        if pred_label[0] == 1:
                            confusion_matrix[1, 0] += 1  # Predicted 'ma'
                        if pred_label[2] == 1:
                            confusion_matrix[1, 2] += 1  # Predicted 'bw'
                if true_label[2] == 1:  # Check if true label is 'bw'
                    if pred_label[2] == 1:
                        confusion_matrix[2, 2] += 1  # True positive for 'bw'
                    else:
                        # False negative for 'bw', update based on predicted class
                        if np.sum(pred_label) == 0:
                            confusion_matrix[2, 3] += 1  # Predicted 'none'
                        if pred_label[0] == 1:
                            confusion_matrix[2, 0] += 1  # Predicted 'ma'
                        if pred_label[1] == 1:
                            confusion_matrix[2, 1] += 1  # Predicted 'em'

    return confusion_matrix.astype(int)



def cm1_8C(cm, y_true, y_pred):
    # Iterate over each class (0 to 7)
    for i in range(8):
        # Initialize confusion matrix components for class i
        TP = FP = TN = FN = 0

        # Iterate over the data
        for true_label, pred_label in zip(y_true, y_pred):
            if true_label == i:
                if pred_label == i:
                    TP += 1  # True Positive
                else:
                    FN += 1  # False Negative
            else:
                if pred_label == i:
                    FP += 1  # False Positive
                else:
                    TN += 1  # True Negative

        # Store the results in the confusion_matrices array
        cm[0, 0, i] = TN
        cm[0, 1, i] = FP
        cm[1, 0, i] = FN
        cm[1, 1, i] = TP

        return cm


def cm8_all(cm, y_true, y_pred):
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1

    return cm

