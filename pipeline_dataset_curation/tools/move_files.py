import os
import shutil
def move_files(ecg_ids, data_dir, split_dir, target_dir):
    for ecg_id in ecg_ids:
        file_name = f"{ecg_id}.npy"
        src_path = os.path.join(data_dir, file_name)
        dest_path = os.path.join(target_dir, file_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
        else:
            print(f"File {src_path} not found.")
    print(f"Data split into train, test, and validation sets and moved to {split_dir}.")