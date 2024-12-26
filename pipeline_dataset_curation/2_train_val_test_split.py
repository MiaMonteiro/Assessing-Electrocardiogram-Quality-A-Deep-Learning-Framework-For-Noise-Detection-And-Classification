import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tools.move_files import move_files


csv_file = r'C:\Users\marci\paper_proj_dataset\ptb-xl_500hz\filtered_ptbxl_database.csv'
# Define the paths of where the csv file and the data are located
data_dir = r'C:\Users\marci\paper_proj_dataset\ptb-xl_500hz'
# Define the path of where the split data will be saved
split_dir = r'C:\Users\marci\paper_proj_dataset\ptb_xl_500_split'

# Read the CSV file
df = pd.read_csv(csv_file)

# Get unique patient IDs
# So that there's repetition of patient in the different sets
unique_patient_ids = df['patient_id'].unique()

# Split patient IDs into train_all_leads, test_all_leads, and validation sets
    # First split: the unique patient IDs are split into 70% train_all_leads and 30% temporary sets
    # Second split: the temporary set is split into 50% test_all_leads and 50% validation sets

# apesar dos ids dos pacientes não estarem aparentemente cronologicamente ordenados no dataset, é recomendável shuffle na mesma.
# pois pode haver alguma ordem implícita que não é visível evitando bias

train_ids, temp_ids = train_test_split(unique_patient_ids, test_size=0.3, random_state=42, shuffle=True)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42, shuffle=True)

# Create directories inside the main split directory for the train, test and validation sets
# Note that one file includes all 12 leads
train_dir = os.path.join(split_dir, 'x_train_all_leads')
test_dir = os.path.join(split_dir, 'x_test_all_leads')
val_dir = os.path.join(split_dir, 'x_val_all_leads')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Assign ecg_ids to each set based on patient_id
train_ecg_ids = df[df['patient_id'].isin(train_ids)]['ecg_id'].tolist()
test_ecg_ids = df[df['patient_id'].isin(test_ids)]['ecg_id'].tolist()
val_ecg_ids = df[df['patient_id'].isin(val_ids)]['ecg_id'].tolist()


# Move files to respective directories
move_files(train_ecg_ids, data_dir, split_dir,train_dir)
move_files(test_ecg_ids, data_dir, split_dir, test_dir)
move_files(val_ecg_ids, data_dir, split_dir, val_dir)


print(f"Number of training patients: {len(train_ids)}")
print(f"Number of validation patients: {len(val_ids)}")
print(f"Number of test patients: {len(test_ids)}")

print(f"Number of training ECGs: {len(train_ecg_ids)}")
print(f"Number of validation ECGs: {len(val_ecg_ids)}")
print(f"Number of test ECGs: {len(test_ecg_ids)}")

# currently each .npy has the shape (5000, 12) which means 5000 samples and 12 leads

# Now we need to split each .npy file into the leads we want meaning the 3 most clean leads


# ############ USERS GUIDEEEE ####################################33
#
#
# # Define the paths for the CSV file and data directory
# # (Update these paths to your local setup)
# csv_file = r'/path/to/your/filtered_ptbxl_database.csv'  # Path to the filtered PTB-XL database CSV
# data_dir = r'/path/to/your/ptb-xl_500hz'                 # Path to the directory containing ECG data files
#
# # Define the directory where the split data will be saved
# split_dir = r'/path/to/your/ptb_xl_360_split'            # Path to the directory where the split data will be stored
#
# # Read the CSV file into a DataFrame
# df = pd.read_csv(csv_file)
#
# # Extract unique patient IDs from the DataFrame
# # This ensures that a single patient does not appear in multiple sets (train, validation, test) removing possible
# # relations between the sets
# unique_patient_ids = df['patient_id'].unique()
#
# # Split patient IDs into training, testing, and validation sets
# # First split: 70% of patient IDs go to the training set, and 30% to a temporary set
# # Second split: The temporary set is split 50/50 into test and validation sets
# # The random_state parameter ensures reproducibility
# # The shuffle parameter ensures that the patient IDs are shuffled before splitting to avoid unintentional bias
# train_ids, temp_ids = train_test_split(unique_patient_ids, test_size=0.3, random_state=42, shuffle=True)
# val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42, shuffle=True)
#
# # Create directories inside the main split directory for the train, test, and validation sets
# # These directories will store the split data files
# train_dir = os.path.join(split_dir, 'train_all_leads')
# test_dir = os.path.join(split_dir, 'test_all_leads')
# val_dir = os.path.join(split_dir, 'val_all_leads')
# os.makedirs(train_dir, exist_ok=True)
# os.makedirs(test_dir, exist_ok=True)
# os.makedirs(val_dir, exist_ok=True)
#
# # Assign ECG IDs to each set based on patient ID
# # This ensures that the ECG data is correctly associated with the appropriate set
# train_ecg_ids = df[df['patient_id'].isin(train_ids)]['ecg_id'].tolist()
# test_ecg_ids = df[df['patient_id'].isin(test_ids)]['ecg_id'].tolist()
# val_ecg_ids = df[df['patient_id'].isin(val_ids)]['ecg_id'].tolist()
#
# # Move the files to their respective directories (train, test, validation)
# # The move_files function handles the actual file transfer
# move_files(train_ecg_ids, data_dir, split_dir, train_dir)
# move_files(test_ecg_ids, data_dir, split_dir, test_dir)
# move_files(val_ecg_ids, data_dir, split_dir, val_dir)
#
# # Print out the number of patients and ECGs in each set
# print(f"Number of training patients: {len(train_ids)}")
# print(f"Number of validation patients: {len(val_ids)}")
# print(f"Number of test patients: {len(test_ids)}")
#
# print(f"Number of training ECGs: {len(train_ecg_ids)}")
# print(f"Number of validation ECGs: {len(val_ecg_ids)}")
# print(f"Number of test ECGs: {len(test_ecg_ids)}")
#
# # Notes on the ECG data format:
# # Each .npy file currently has the shape (5000, 12), which represents 5000 samples across 12 leads
#
# # Next steps involve splitting each .npy file into separate files for individual leads and storing the 3 most clean
# leads