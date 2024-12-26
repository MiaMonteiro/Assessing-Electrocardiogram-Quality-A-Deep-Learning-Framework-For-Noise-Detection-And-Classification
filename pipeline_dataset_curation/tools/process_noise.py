from sklearn.preprocessing import minmax_scale
import numpy as np
from scipy.stats import zscore


def process_noise(noise):

    # Concatenate the two channels so that the noise data is 1D (2 channels)
    concatenated = np.concatenate((np.array(noise[:, 0]), np.array(noise[:, 1])))
    zscored = zscore(concatenated) # Z-score normalization
    # normalized = minmax_scale(zscored) # Min-Max scaling
    return zscored
