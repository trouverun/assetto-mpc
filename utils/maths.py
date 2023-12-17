import numpy as np
import scipy
import matplotlib.pyplot as plt


def rotate_around(center, angle, values):
    angle = -(angle - np.deg2rad(90))#angle + np.deg2rad(90)

    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    return (R @ (values - center).T).T


def save_spectrum_plot(data, fs, path):
    """
    Save a frequency spectrum plot of the provided data.

    Parameters:
    - data: The input data array.
    - fs: The sampling frequency of the data.
    - path: Path (including filename) where the plot should be saved.
    """

    # Compute the FFT of the data
    spectrum = np.fft.fft(data)

    # Convert magnitude to dB scale and normalize
    magnitude_dB = 20 * np.log10(np.abs(spectrum / len(data)))

    # Frequency axis (upto Nyquist frequency)
    freqs = np.linspace(0, fs / 2, len(data) // 2)

    # Plot the magnitude in dB of the FFT (only the positive frequencies)
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, magnitude_dB[:len(data) // 2])
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)

    # Save the plot to the specified path
    plt.savefig(path)
    plt.close()



def zero_phase_filtering(data, cutoff_freq, fs, window_size=1000, order=4):
    """
    Zero-phase Butterworth filtering using a sliding window approach.

    Parameters:
    - data: The input data array.
    - cutoff_freq: The cutoff frequency for the Butterworth filter.
    - fs: The sampling frequency of the data.
    - window_size: Size of the sliding window.
    - order: The order of the Butterworth filter.

    Returns:
    - filtered_data: The filtered data.
    """

    # Create Butterworth filter coefficients
    b, a = scipy.signal.butter(order, cutoff_freq / (0.5 * fs), btype='low')

    # Number of valid points from each window
    valid_points = window_size // 2

    # Pre-allocate memory for filtered data
    filtered_data = np.zeros(len(data) - window_size + valid_points)

    for i in range(0, len(data) - window_size + 1):
        # Extract the current window from data
        window = data[i:i + window_size]

        # Forward filtering
        forward_filtered = scipy.signal.lfilter(b, a, window)

        # Backward filtering
        backward_filtered = scipy.signal.lfilter(b, a, forward_filtered[::-1])[::-1]

        # Extract valid points and store in filtered data
        filtered_data[i:i + valid_points] = backward_filtered[window_size // 4:window_size // 4 + valid_points]

    return filtered_data

