
import os
import shutil

# Define the folder paths
source_clean = r'C:\Users\marci\paper_proj_dataset\ptb_xl_noisy_360_normalized\x_test_noisy'
destination_clean = r'C:\Users\marci\paper_proj_dataset\x_test_noisy_order_fixed'

# create the dir if does not exist

if not os.path.exists(destination_clean):
    os.makedirs(destination_clean)

# read the signals in test selected leads
clean_signal_files = [f for f in os.listdir(source_clean) if f.endswith('.npy')]
# sort them in order
sorted_files = sorted(clean_signal_files, key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1].split('.')[0])))

# loop through the files and copy them to the destination folder with new names
for new_index, old_file in enumerate(sorted_files):


    new_file_name = f"{new_index}.npy"

    # build the full paths
    old_file_path = os.path.join(source_clean, old_file)
    new_file_path = os.path.join(destination_clean, new_file_name)

    shutil.copyfile(old_file_path, new_file_path)

    print(f"Copied {old_file} to {new_file_name} in {destination_clean}")