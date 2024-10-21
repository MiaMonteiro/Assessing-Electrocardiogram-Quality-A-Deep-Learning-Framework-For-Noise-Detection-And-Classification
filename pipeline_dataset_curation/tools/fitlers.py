# this has the intention of clean up the signals without destryoing the original signal significantly
# this way i can make the selection of the records that are really noisy in 2_train_val_test_split.py

# imports

import numpy as np
import os
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.preprocessing import minmax_scale
from scipy.signal import resample


# Define the base directory and directories to check

def smooth(y, window):
    wind = np.ones(window) / window
    y_smooth = np.convolve(y, wind, mode='same')

    return y_smooth


def filtering(signal, fs = 360):

    nyquist = fs / 2

    # Band-pass filter parameters for BW
    lowcut_bp = 1 / nyquist
    highcut_bp = 45 / nyquist
    # using filtfilt
    b_bp, a_bp = butter(3, [lowcut_bp, highcut_bp], btype='band', output='ba')

    # Apply the band-pass filter
    filtered_signal_bp = filtfilt(b_bp, a_bp, signal)
    # filtered_signal_bp_notch = filtfilt(b_notch, a_notch, filtered_signal_bp)

    padding_size = 7 // 2
    filtered_signal_bp = np.pad(filtered_signal_bp, (padding_size, padding_size), mode='reflect')
    # print('filtered_signal after bp',filtered_signal_bp.shape)
    filtered_signal_bp = smooth(filtered_signal_bp, 7)
    # print('filtered_signal after smooth',filtered_signal_bp.shape)
    filtered_signal_bp_smooth = filtered_signal_bp[padding_size:-padding_size]
    # print('filtered_signal after padding',filtered_signal_bp_smooth.shape)
    # final_filtered_signal = minmax_scale(filtered_signal_bp_smooth)

    return filtered_signal_bp_smooth


