import os
import shutil

# TODO: Add a function that allows to empty only a folder

def ensure_folders_empty(folders):
    """
    Ensures that specified folders are empty. If a folder already exists, it will be deleted and then recreated.

    This function takes a list of folder paths, checks if each folder exists, and if so, deletes it along with its contents. After deletion, it recreates the folder to ensure it is empty and ready for new data.

    Args:
        folders (list of str): List of folder paths to check and ensure are empty.

    Returns:
        None: This function does not return any value. It performs operations directly on the file system.

    Example:
        ensure_folders_empty(['/path/to/folder1', '/path/to/folder2'])

    Notes:
        - Be cautious when using this function as it will permanently delete all contents of the specified folders.
        - If a folder does not exist, it will simply be created.
    """
    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)