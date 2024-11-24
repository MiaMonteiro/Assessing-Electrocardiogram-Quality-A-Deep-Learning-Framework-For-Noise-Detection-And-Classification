import numpy as np
import os
import importlib.util
import sys

sqis_path = r'C:\Users\marci\signal_quality\signal_quality\sqis.py'  # Adjust this path

# Load the module
spec = importlib.util.spec_from_file_location("sqis", sqis_path)
sqis = importlib.util.module_from_spec(spec)
sys.modules["sqis"] = sqis
spec.loader.exec_module(sqis)


noisy_folder = r'C:\Users\marci\paper_proj_dataset\x_test_clean_order_fixed'
clean_folder = r'C:\Users\marci\paper_proj_dataset\x_test_noisy_order_fixed'

def calculate_snr_power_based(clean_signal, noisy_signal):
    # Implement the SNR calculation
    power_clean = np.sum(clean_signal ** 2)
    power_noise = np.sum((noisy_signal - clean_signal) ** 2)
    return 10 * np.log10(power_clean / power_noise)


file_name = '183.npy'
clean_signal = np.load(os.path.join(clean_folder, file_name))
noisy_signal = np.load(os.path.join(noisy_folder, file_name))

# snr = calculate_snr_power_based(clean_signal, noisy_signal)
# kurt = sqis.k_sqi(noisy_signal, kurtosis_method='fisher')
# psd = sqis.p_sqi(noisy_signal, 360, 5)
# bas = sqis.bas_sqi(noisy_signal, 360, 5)
# skew = sqis.s_sqi(noisy_signal)



# print("Metrics for file:", file_name)
# print("Kurtosis:", kurt)
# print("Skewness:", skew)
# print("SNR:", snr)
# print("PSD:", psd)
# print("BAS:", bas)

import neurokit2 as nk

# zhao methods
q = nk.ecg_quality(noisy_signal, sampling_rate=360, method='zhao2018')
print("Zhao method:", q)

