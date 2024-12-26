import os
import numpy as np

# Define the folder path where noise info .npy files are stored
noise_info_folder = r'C:\Users\marci\paper_proj_dataset\ptb_xl_noisy\noise_info\x_test_noise_info'



# Iterate through files in the noise info folder
for file_name in os.listdir(noise_info_folder):
    if file_name.endswith(".npy"):
        file_path = os.path.join(noise_info_folder, file_name)
        # Load the noise info from the .npy file
        noise_info = np.load(file_path, allow_pickle=True)
        print(noise_info)
        # # Print information about the loaded noise info
        # print('-----------------------------------------------')
        # print(f"File name: {file_name}")
        # print("Noise info:")
        # for noise in noise_info:
        #     start_point = noise[0]
        #     end_point = noise[1]
        #     ma = noise[2]
        #     em = noise[3]
        #     bw = noise[4]
        #     print(f"  Start: {start_point}, End: {end_point} | MA: {ma}, EM: {em}, BW: {bw}")