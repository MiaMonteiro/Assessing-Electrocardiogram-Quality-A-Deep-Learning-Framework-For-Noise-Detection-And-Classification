import os
from tools.process_leads_clean import track_and_select_clean_leads

# currently each .npy has the shape (5000, 12) which means 5000 samples and 12 leads
# we want to keep only 3 leads with the fewest peaks after filtering and resampling

split_dir = r'C:\Users\marci\paper_proj_dataset\ptb_xl_500_split'

train_dir = os.path.join(split_dir, 'x_train_all_leads')
test_dir = os.path.join(split_dir, 'x_test_all_leads')
val_dir = os.path.join(split_dir, 'x_val_all_leads')

processed_dir = r'C:\Users\marci\paper_proj_dataset\ptb_xl_clean_360'
os.makedirs(processed_dir, exist_ok=True)

# Example usage
csv_path = r'C:\Users\marci\paper_proj_dataset\ptb-xl_500hz\filtered_ptbxl_database.csv'

# Output CSV paths
output_csv_test = r'C:\Users\marci\paper_proj_dataset\ptb_xl_clean_360\x_test_clean_ptbxl_database.csv'
output_csv_train = r'C:\Users\marci\paper_proj_dataset\ptb_xl_clean_360\x_train_clean_leads_ptbxl_database.csv'
output_csv_val = r'C:\Users\marci\paper_proj_dataset\ptb_xl_clean_360\x_val_clean_ptbxl_database.csv'

# Process and track signals for each set

track_and_select_clean_leads(csv_path, test_dir, os.path.join(processed_dir, 'x_test_clean'), output_csv_test)
track_and_select_clean_leads(csv_path, train_dir, os.path.join(processed_dir, 'x_train_clean'), output_csv_train)
track_and_select_clean_leads(csv_path, val_dir, os.path.join(processed_dir, 'x_val_clean'), output_csv_val)
