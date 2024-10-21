import numpy as np


def snr(clean_signal, noisy_signal):
    power_clean = np.sum(clean_signal ** 2)
    power_noise = np.sum((noisy_signal - clean_signal) ** 2)

    # Cap se o power of noise is negative to 0
    # power_noise = max(power_noise, 1e-10)
    #
    # # Return SNR as infinte if the power_noise is zero to avoid divide-by-zero
    # if power_noise == 0:
    #     return float('inf')
    return 10 * np.log10(power_clean / power_noise), power_noise, power_clean



def scale_snr(snr_value, power_noise, power_clean, max_snr=40):
    if snr_value == float('inf'):
        snr_value = 1  # or cap it to a maximum if desired
    if power_noise < power_clean:
        snr_value = 0
    scaled_value = np.log10(snr_value + 1) / np.log10(max_snr + 1)
    return round(min(scaled_value, 1), 3)