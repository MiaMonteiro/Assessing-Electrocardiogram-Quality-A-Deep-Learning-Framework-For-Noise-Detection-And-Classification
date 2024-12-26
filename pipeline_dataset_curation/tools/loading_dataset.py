import pandas as pd
from tqdm import tqdm
import wfdb
import os
import numpy as np


def preprocess_csv(csv_file_path, csv_filename, output_dir):
    """
    Preprocesses the CSV file by filtering out records with quality issues and updating the ECG ID column.

    Args:
        csv_file_path (str): Path to the directory containing the CSV file.
        csv_filename (str): Name of the CSV file to be processed.
        output_dir (str): Directory where the processed CSV file will be saved.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered records with updated ECG IDs.
    """
    csv_file = os.path.join(csv_file_path, csv_filename)
    print(f"Reading CSV file: {csv_filename}")
    df = pd.read_csv(csv_file)

    print("CSV file read successfully.")

    # Define the columns to check for issues
    cols_to_check = ['baseline_drift', 'static_noise', 'burst_noise', 'electrodes_problems']

    # Create a dictionary to store quality issues
    quality_issues = {}

    for index, row in df.iterrows():
        filename_lr = row['filename_lr']
        filename_hr = row['filename_hr']

        # Check if there are any quality issues for filename_lr
        has_issue_lr = any(isinstance(row[col], str) and row[col].strip() for col in cols_to_check)

        # Check if there are any quality issues for filename_hr
        has_issue_hr = any(isinstance(row[col], str) and row[col].strip() for col in cols_to_check)

        # Store in dictionary
        if filename_lr:
            quality_issues[filename_lr] = has_issue_lr
        if filename_hr:
            quality_issues[filename_hr] = has_issue_hr

    # Create a boolean mask to filter rows with quality issues
    mask = df.apply(lambda row: not (quality_issues.get(row['filename_lr'], False) or quality_issues.get(row['filename_hr'], False)), axis=1)

    # Apply the mask to filter out rows
    df_filtered = df[mask].copy()  # Ensure we're working with a copy of the filtered DataFrame

    # Reset index of df_filtered
    df_filtered.reset_index(drop=True, inplace=True)

    # Update ecg_id column to start from 0
    df_filtered.loc[:, 'ecg_id'] = range(len(df_filtered))

    # Define the path to save the filtered CSV
    filtered_csv_path = os.path.join(output_dir, 'filtered_ptbxl_database.csv')

    # Save the filtered DataFrame to a new CSV file
    print(f"Saving filtered DataFrame to {filtered_csv_path}")
    df_filtered.to_csv(filtered_csv_path, index=False)
    print("Filtered DataFrame saved successfully.")

    return df_filtered


def load_raw_data_local(rec_file, csv_file_path, csv_filename, local_dir,  output_dir, sr=500):
    """
    Loads raw ECG data from local files based on the sampling rate and filters the data using the preprocessed CSV.

    Args:
        rec_file (str): Path to the file listing all ECG records.
        csv_file_path (str): Path to the directory containing the CSV file.
        csv_filename (str): Name of the CSV file.
        local_dir (str): Directory where the raw ECG data is stored.
        output_dir (str): Directory where the filtered CSV file will be saved.
        sr (int, optional): Sampling rate for filtering the records. Defaults to 500 Hz.

    Returns:
        list: A list containing the loaded ECG data as numpy arrays.
    """

    # Call preprocess_csv to filter and save the CSV
    filtered_df = preprocess_csv(csv_file_path, csv_filename, output_dir)

    rec_prefix = f'records{sr}'
    filtered_filenames = []

    for index, row in filtered_df.iterrows():
        if sr == 100 and row['filename_lr'].startswith(rec_prefix):
            filtered_filenames.append(row['filename_lr'])
        elif sr == 500 and row['filename_hr'].startswith(rec_prefix):
            filtered_filenames.append(row['filename_hr'])

    print(f'There are a total of {len(filtered_filenames)} records with a sample rate of {sr} Hz.')

    skipped_files = 0
    data = []

    for f in tqdm(filtered_filenames, desc="Loading records"):
        filepath = os.path.join(local_dir, f)
        # print('filepath:', filepath)
        try:
            content, meta = wfdb.rdsamp(filepath)
            data.append(content)
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            skipped_files += 1

    print(f'Skipped {skipped_files} files due to errors during loading.')
    return data

def ptbxl_save(data, save_dir='data500'):
    """
    Saves each ECG record from the data list as a separate numpy file.

    Args:
        data (list): List containing ECG records as numpy arrays.
        save_dir (str, optional): Directory where the numpy files should be saved. Defaults to 'data500'.

    Returns:
        None: Saves the numpy files to the specified directory.
    """
    for i in range(np.shape(data)[0]):
        np.save(save_dir + '/' + str(i) + '.npy', data[i])
