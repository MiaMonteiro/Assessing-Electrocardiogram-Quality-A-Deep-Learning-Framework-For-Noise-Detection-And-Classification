import gc
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
from plot_metrics_model import plot_loss_curves
from model_tracking import update_excel
from eval_metrics import cm_4classes, cm_1class, calculate_metrics
from plot_metrics_model import plot_cm4, plot_cm1
import warnings

# import random
# from torch.utils.data import Subset


class ECGDataset(Dataset):

    def __init__(self, main_dir, subset):
        """
        Initialize the dataset with the directory containing the data.

        :param main_dir: Main directory where the ECG data is stored
        :param subset: Subset of data ('train', 'val', etc.)
        """
        # Set paths for input (X) and labels (Y)
        self.X_dir = os.path.join(main_dir, f'x_{subset}')
        print(f'path{self.X_dir}')
        self.Y_dir = os.path.join(main_dir, f'y_{subset}')
        # self.clean_dir = os.path.join(main_dir, f'clean_{subset}')  # Clean signals
        # List of signal files to be loaded
        self.signals = [f for f in os.listdir(self.X_dir) if f.endswith('.npy')]
        print(f"Number of .npy files in the {subset} directory: {len(self.signals)}")

    def __len__(self):
        """
        Returns the total number of samples in the dataset
        """
        return len(self.signals)

    def __getitem__(self, idx):
        """
        Retrieves a sample and its label from the dataset, along with the clean signal.

        :param idx: Index of the sample to retrieve
        :return: Tuple (x, y, clean) where:
                 - x is the noisy signal
                 - y is the label
                 - clean is the clean signal
        """
        # Load the input signal and label
        x = np.load(os.path.join(self.X_dir, self.signals[idx]))
        y = np.load(os.path.join(self.Y_dir, self.signals[idx]))
        # clean = np.load(os.path.join(self.clean_dir, self.signals[idx]))  # Clean signal
        # print(idx)
        # print(f"Shape of x before reshaping (dataset): {x.shape}")
        # Convert numpy arrays to torch tensors
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()  # y should be in the shape (seq_len, num_classes)
        # clean = torch.from_numpy(clean).float()


        x = x.reshape(-1, 1)  # Ensure x is of shape (seq_len, input_size) which already is???
        # clean= clean.reshape(-1, 1)
        # print(f"Shape of x after reshaping (dataset): {x.shape}")
        # Shape of x before reshaping(dataset): (3600,)
        # Shape of x after reshaping(dataset): torch.Size([3600, 1])

        return x, y



class ONEHOTgru(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_states, bidirectional=True, dropout=0.5):
        super(ONEHOTgru, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=(dropout if num_layers > 1 else 0))
        self.hidden_size = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hidden_size, num_states)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        # gru_out = self.dropout(gru_out)  # remove external dropout
        output = self.fc(gru_out)
        return output



# print("Shape of x:", x.shape) Shape of x: torch.Size([32, 3600, 1]) batch_size, seq_len, input_size
# print("Shape of y:", y.shape) [32, 3600, 3] batch_size, seq_len, num_states
# print(f"Example target: {y[0, :10]}")  # Print a subset of the target
# print("Shape of outputs:", outputs.shape) [32, 3600, 3] batch_size, seq_len, num_states
# print(f"Example output: {outputs[0, :10]}")  # Print a subset of the output

def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0

    with tqdm(total=len(train_loader), desc=f'Training Epoch {epoch + 1}') as pbar:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)

            outputs = outputs.view(-1, outputs.size(-1))
            y = y.view(-1, y.size(-1))

            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            del x, y, outputs
            torch.cuda.empty_cache()

            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

    avg_train_loss = running_loss / len(train_loader)

    print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.6f}")

    # Clean up memory
    torch.cuda.empty_cache()
    gc.collect()

    return avg_train_loss


def validate(model, val_loader, criterion, epoch):

    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0

    with tqdm(total=len(val_loader), desc=f'Validating Epoch {epoch + 1}') as pbar:
        with torch.no_grad():  # Disable gradient computation for validation
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)

                # Flatten outputs and targets for compatibility with loss function
                outputs = outputs.view(-1, outputs.size(-1))
                y = y.view(-1, y.size(-1))

                loss = criterion(outputs, y)  # Calculate loss
                val_loss += loss.item()  # Accumulate validation loss


                # Cleanup unnecessary variables
                del x, y, outputs
                torch.cuda.empty_cache()

                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.6f}")

    # Clean up memory
    torch.cuda.empty_cache()
    gc.collect()

    return avg_val_loss

def test_model(model_path, device, test_loader, idx_samples_to_save):

    checkpoint = torch.load(model_path)

    # Initialize the model architecture
    model = ONEHOTgru(
        input_size=checkpoint['hyperparameters']['input_size'],
        hidden_size=checkpoint['hyperparameters']['hidden_size'],
        num_layers=checkpoint['hyperparameters']['num_layers'],
        num_states=checkpoint['hyperparameters']['num_states'],
        bidirectional=checkpoint['hyperparameters']['bidirectional']
    ).to(device)

    # Load the saved state_dict into the model
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()  # Set the model to evaluation mode

    all_preds = []
    all_labels = []
    all_inputs = []

    # Inference with progress bar
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Processing test data", unit="batch"):
            inputs, labels = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).int()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_inputs.append(inputs.cpu().numpy())

    # Concatenate all predictions and labels
    all_inputs = np.concatenate(all_inputs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate the confusion matrix for the 4 classes
    cm4_zeros = np.zeros((4, 4), dtype=int)
    cm4 = cm_4classes(cm4_zeros, all_labels, all_preds)

    # Plot the confusion matrix for the individual classes
    cm1 = np.zeros((2, 2, 3), dtype=int)
    cm1 = cm_1class(cm1, all_labels, all_preds)

    # Select the corresponding inputs, labels, and predictions
    inputs_to_save = all_inputs[idx_samples_to_save]
    labels_to_save = all_labels[idx_samples_to_save]
    preds_to_save = all_preds[idx_samples_to_save]

    del all_inputs, all_labels, all_preds
    torch.cuda.empty_cache()

    return cm4, cm1, inputs_to_save, preds_to_save, labels_to_save


def main(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        num_states=3,
        bidirectional=False,
        dropout=0,
        learning_rate=0.001,
        batch_size=128,
        num_epochs=200,
        patience=40,
        main_dir=r'C:\Users\marci\paper_proj_dataset\ptb_xl_final'):

    global device

    warnings.filterwarnings("ignore", category=FutureWarning)


    # Select the device to use for training (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed=42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Get the current date and time for organizing output directories
    now = datetime.now()
    current_time = now.strftime("%d%m_%H%M")  # Format: d%m_%H%M (day, month, hour, minute)

    # Define metadata for saving model outputs using the date and time
    metadata = f'ONEHOT_GRU_layers{num_layers}_hiddensize_{hidden_size}_states{num_states}_patience{patience}_biderectional{bidirectional}_drop{dropout}_date{current_time}'
    base_output_dir = '.\model_predictions'
    output_dir = os.path.join(base_output_dir, metadata)
    os.makedirs(output_dir, exist_ok=True)

    model_name = '_'.join(metadata.split('_')[:2])

    print('Starting the training process...')
    print(f'Using device: {device}')
    print(f'Output directory: {output_dir}')
    # print(f"Training with the following hyperparameters: {args}")

    # Create training and validation datasets and loaders
    train_dataset = ECGDataset(main_dir=main_dir, subset='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False, prefetch_factor=2)
    print(f"Number of training samples: {len(train_dataset)}")

    val_dataset = ECGDataset(main_dir=main_dir, subset='val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False, prefetch_factor=2)
    print(f"Number of validation samples: {len(val_dataset)}")

    test_dataset = ECGDataset(main_dir=main_dir, subset='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False, prefetch_factor=2)
    print(f"Number of test samples: {len(test_dataset)}")

    # Initialize the model, loss function, and optimizer
    model = ONEHOTgru(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_states=num_states,
                      bidirectional=bidirectional).to(device)

    # Compute class weights based on class frequencies to handle class imbalance
    counts = [23920072, 24092161, 41116619]  # Counts for MA, EM, BW
    total_counts = sum(counts)
    frequencies = [count / total_counts for count in counts]
    class_weights = [1 / freq for freq in frequencies]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Set up the loss function with class weights
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')  # Initialize the best validation loss
    best_epoch = 0
    epochs_no_improve = 0  # Early stopping counter

    # Initialize variables to store loss values
    train_loss = None
    val_loss = None
    epoch = 0
    # Initialize lists to store loss values for plotting
    train_losses = []
    val_losses = []

    # Train the model across the specified number of epochs
    for epoch in range(num_epochs):
        # Training step
        train_loss = train(model, train_loader, criterion, optimizer, epoch)
        train_losses.append(train_loss)

        # Validation step
        val_loss = validate(model, val_loader, criterion, epoch)
        val_losses.append(val_loss)

        # Check if the validation loss is the best we've seen so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0

            # Save the model if it has improved
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'training_loss': train_loss,
                'val_loss': val_loss,
                'hyperparameters': {
                    'input_size': input_size,
                    'num_layers': num_layers,
                    'hidden_size': hidden_size,
                    'num_states': num_states,
                    'bidirectional': bidirectional,
                    'dropout': dropout,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'num_epochs': num_epochs,
                    'patience': patience,
                    'class_weights': class_weights
                }
            }, os.path.join(output_dir, 'best_model.pth'))

            print(f"Best model saved at epoch {epoch + 1} with validation loss: {best_val_loss:.6f}")

        else:
            epochs_no_improve += 1

        # Early stopping logic
        if epochs_no_improve >= patience:
            print(
                f"Early stopping at epoch {epoch + 1}. No improvement in validation loss for {patience} consecutive epochs.")
            break

    # Save the final model after training completes or early stopping is triggered
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'training_loss': train_loss,
        'val_loss': val_loss,
        'hyperparameters': {
            'input_size': input_size,
            'num_layers': num_layers,
            'hidden_size': hidden_size,
            'num_states': num_states,
            'bidirectional': bidirectional,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'patience': patience,
            'class_weights': class_weights
        }
    }, os.path.join(output_dir, 'final_model.pth'))

    print(
        f"Final model saved after {epoch + 1} epochs. Best epoch: {best_epoch + 1} with validation loss: {best_val_loss:.6f}")

    plot_loss_curves(train_losses, val_losses, output_dir)
    print("Loss curves plot saved.")

    # Run the test model function

    all_idx_test = len(test_loader.dataset)
    # Ensure consistent sample selection
    idx_samples_to_save = np.random.choice(all_idx_test, 100, replace=False)

    print("Starting the testing process...")
    # Run the test model function

    model_path = os.path.join(output_dir,'final_model.pth')
    cm4, cm1, inputs_to_save, preds_to_save, labels_to_save = test_model(model_path, device, test_loader,
                                                                         idx_samples_to_save)
    print('Saving the selected inputs, labels, and predictions...')
    # Save the selected inputs, labels, and predictions
    np.save(os.path.join(output_dir, 'inputs.npy'), inputs_to_save)
    np.save(os.path.join(output_dir, 'labels.npy'), labels_to_save)
    np.save(os.path.join(output_dir, 'preds.npy'), preds_to_save)
    print('Inputs, labels, and predictions saved.')

    # Plot the confusion matrices
    print('Plotting the confusion matrices...')
    plot_cm1(cm1, output_dir)
    print('Confusion matrices CM1 saved.')
    plot_cm4(cm4, output_dir)
    print('Confusion matrices CM4 saved.')

    print('Calculating the metrics for the individual classes...')
    # Calculate the metrics for the individual classes
    metrics = calculate_metrics(cm1)

    for class_name, class_metrics in metrics.items():
        print(f"Metrics for {class_name}:")
        for metric_name, value in class_metrics.items():
            print(f"  {metric_name}: {value:.2f}")

    print('Model testing process completed.')

    print('Updating tracking Excel...')

    best_epoch_corrected = best_epoch + 1
    excel_file_path = os.path.join(base_output_dir, 'model_results.xlsx')
    update_excel(excel_file_path, model_name, input_size, hidden_size, num_layers, num_states,
    bidirectional, dropout, learning_rate, batch_size, num_epochs, patience, train_loss, best_val_loss, best_epoch_corrected)
    print('Excel file updated with the latest model results.')

if __name__ == '__main__':

    main()