import numpy as np
import pickle
import os
from tools.loading_dataset import load_nstdb_raw, load_raw_data_local, ptbxl_save
import wfdb
from matplotlib import pyplot as plt
from sklearn.preprocessing import minmax_scale
from scipy.signal import resample

# to do: Remove the option to load directly from the Physionet database AND keep the local downloads only for simplicity


# This script is used to load the PTB-XL database and save it as numpy arrays
# Because of x y z the loading of the ECGs needs to be from local downloaded files
# Also includes the loading of the MIT-BIH noise database directly from Physionet

# define paths for the list of the records and the csv file
rec_file = r'C:\Users\marci\paper_proj_dataset\ptb-xl\RECORDS'
csv_file_path = r'C:\Users\marci\paper_proj_dataset\ptb-xl'
csv_filename = 'ptbxl_database.csv'

dir = r'C:\Users\marci\paper_proj_dataset\ptb-xl_500hz'
# make directory if it doesn't exist
os.makedirs(dir, exist_ok=True)
# path to the local folder
local_dir = r'C:\Users\marci\paper_proj_dataset\ptb-xl'

data = load_raw_data_local(rec_file, csv_file_path, csv_filename, local_dir, dir, sr=500)
data = np.array(data)

# save each record separately as a numpy array
ptbxl_save(data, dir)

# load MIT-BIH noise database direclty from Physionet
# pn_dir_noise = 'nstdb'
# data_noise = load_nstdb_raw(pn_dir_noise)

# Or download the dataset and load them from a local folder.
dir_noise = r'C:\Users\marci\paper_proj_dataset\paper\mit_bih_noise_stress'
bw_filepath = os.path.join(dir_noise, 'ma')
bw, bw_meta = wfdb.rdsamp(bw_filepath)
ma_filepath = os.path.join(dir_noise, 'ma')
ma, ma_meta = wfdb.rdsamp(ma_filepath)
em_filepath = os.path.join(dir_noise, 'em')
em, em_meta = wfdb.rdsamp(em_filepath)

# save .npy
np.save(os.path.join(dir_noise, 'bw'), bw)
np.save(os.path.join(dir_noise, 'ma'), ma)
np.save(os.path.join(dir_noise, 'em'), em)




# ##################### USER GUIDE ############################
#
#
# # This script processes and saves ECG data from the PTB-XL database as numpy arrays from local files.
# # The download of the PTB-XL database from Physionet is available through https://physionet.org/content/ptb-xl/1.0.3/
# # The loading of the MIT-BIH Noise Stress Test Database (NSTDB) is done directly from Physionet therefore doesn't
# # require downloading the files prior. The dataset can be accessed through https://physionet.org/content/nstdb/1.0.0/
#
# # Paths to PTB-XL database files (update these paths to your local setup)
# rec_file = r'/path/to/your/ptb-xl/RECORDS'  # Path to the file listing all ECG records
# csv_file_path = r'/path/to/your/ptb-xl'     # Directory containing the PTB-XL CSV file
# csv_filename = 'ptbxl_database.csv'         # Name of the PTB-XL CSV file
#
# # Define directory to save the processed PTB-XL data
# dir = r'/path/to/your/output/ptb-xl_500hz'
# # Create the directory if it doesn't exist
# os.makedirs(dir, exist_ok=True)
#
# # Local directory for the PTB-XL data
# local_dir = r'/path/to/your/ptb-xl'         # Directory where raw PTB-XL data is stored
#
# # Load the raw PTB-XL data from local files with a specified sampling rate (sr=500 Hz)
# data = load_raw_data_local(rec_file, csv_file_path, csv_filename, local_dir, dir, sr=500)
# data = np.array(data)  # Convert the loaded data into a numpy array for easier manipulation
#
# # Save each ECG record as a separate numpy file in the specified directory
# ptbxl_save(data, dir)
#
# # Load the MIT-BIH Noise Stress Test Database (NSTDB) directly from Physionet
# pn_dir_noise = 'nstdb'  # Identifier for the Physionet NSTDB data
# data_noise = load_nstdb_raw(pn_dir_noise)
#
# # Save the loaded noise data as a pickle file for easy access in future processing
# pickle_out = open("data_noise.pickle", "wb")
# pickle.dump(data_noise, pickle_out)  # Write the noise data to a pickle file
# pickle_out.close()  # Close the pickle file to ensure data integrity
