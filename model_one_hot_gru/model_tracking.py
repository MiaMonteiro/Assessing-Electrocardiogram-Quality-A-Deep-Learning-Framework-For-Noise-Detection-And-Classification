import pandas as pd
import os

def update_excel(excel_file_path, model_name, input_size, hidden_size, num_layers, num_states,
    bidirectional, dropout, learning_rate, batch_size, num_epochs, patience, train_loss, best_val_loss, best_epoch):
    data = {
        'Model Name': model_name,
        'Input Size': input_size,
        'Hidden Size': hidden_size,
        'Num Layers': num_layers,
        'Num States': num_states,
        'Bidirectional': bidirectional,
        'Dropout': dropout,
        'Learning Rate': learning_rate,
        'Batch Size': batch_size,
        'Num Epochs': num_epochs,
        'Patience': patience,
        'Train Loss': train_loss,
        'Best Val Loss': best_val_loss,
        'Best Epoch': best_epoch,
    }

    # for class_name, class_metrics in metrics.items():
    #     for metric_name, value in class_metrics.items():
    #         column_name = f'{class_name} {metric_name}'
    #         data[column_name] = value

    # Convert the data dictionary into a DataFrame
    new_df = pd.DataFrame([data])

    # Check if the Excel file already exists
    if os.path.exists(excel_file_path):
        # Load the existing Excel file
        existing_df = pd.read_excel(excel_file_path)
        # Append the new data
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        # If the file doesn't exist, the new data is the updated data
        updated_df = new_df

    # Save the updated DataFrame back to the Excel file
    updated_df.to_excel(excel_file_path, index=False)


def update_excel8C(excel_file_path, model_name, input_size, hidden_size, num_layers, num_classes, fc, bidirectional,
                   dropout, learning_rate, batch_size, num_epochs, patience, best_epoch,
                   train_loss, best_val_loss):
    data = {
        'Model Name': model_name,
        'Input Size': input_size,
        'Hidden Size': hidden_size,
        'Num Layers': num_layers,
        'Num Classes': num_classes,
        'FC': fc,
        'Bidirectional': bidirectional,
        'Dropout': dropout,
        'Learning Rate': learning_rate,
        'Batch Size': batch_size,
        'Num Epochs': num_epochs,
        'Patience': patience,
        'Train Loss': train_loss,
        'Best Val Loss': best_val_loss,
        'Best Epoch': best_epoch,
    }

    # for class_name, class_metrics in metrics.items():
    #     for metric_name, value in class_metrics.items():
    #         column_name = f'{class_name} {metric_name}'
    #         data[column_name] = value

    # Convert the data dictionary into a DataFrame
    new_df = pd.DataFrame([data])

    # Check if the Excel file already exists
    if os.path.exists(excel_file_path):
        # Load the existing Excel file
        existing_df = pd.read_excel(excel_file_path)
        # Append the new data
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        # If the file doesn't exist, the new data is the updated data
        updated_df = new_df

    # Save the updated DataFrame back to the Excel file
    updated_df.to_excel(excel_file_path, index=False)


