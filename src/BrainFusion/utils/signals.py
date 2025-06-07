# -*- coding: utf-8 -*-
# @Time    : 2024/6/12 14:40
# @Author  : XXX
# @Site    : 
# @File    : signals.py
# @Software: PyCharm 
# @Comment :
import numpy as np
import pywt
from scipy.signal import resample_poly, butter, filtfilt, welch
from scipy.signal import detrend
from scipy.interpolate import UnivariateSpline


def resample_signal(signal, original_fs, target_fs):
    """
    Resample signal from original to target sampling frequency.

    :param signal: Input signal array
    :type signal: np.ndarray
    :param original_fs: Original sampling frequency
    :type original_fs: int
    :param target_fs: Target sampling frequency
    :type target_fs: int
    :return: Resampled signal array
    :rtype: np.ndarray
    """
    # Calculate resampling factors
    gcd = np.gcd(original_fs, target_fs)
    up = target_fs // gcd
    down = original_fs // gcd

    # Apply anti-aliasing filter
    nyquist_rate = 0.5 * original_fs
    cutoff = 0.5 * target_fs
    b, a = butter(4, cutoff / nyquist_rate)
    signal_filtered = filtfilt(b, a, signal)

    # Perform resampling
    return resample_poly(signal_filtered, up, down)


def compute_psd(signal, fs=1000, nperseg=1000, fl=1, fh=40):
    """
    Compute average power spectral density in frequency band.

    :param signal: Input signal array
    :type signal: np.ndarray
    :param fs: Sampling frequency
    :type fs: int
    :param nperseg: Segment length for Welch's method
    :type nperseg: int
    :param fl: Lower frequency bound
    :type fl: int
    :param fh: Upper frequency bound
    :type fh: int
    :return: Average PSD value in band
    :rtype: float
    """
    f, Pxx = welch(signal, fs=fs, nperseg=nperseg)
    return np.mean(Pxx[(f >= fl) & (f <= fh)])


def compute_mean(signal, window_size=None):
    """
    Compute signal mean value with optional windowing.

    :param signal: Input signal array
    :type signal: np.ndarray
    :param window_size: Size of analysis window
    :type window_size: int or None
    :return: Mean values
    :rtype: list or float
    """
    if window_size:
        return [np.mean(signal[i:i + window_size]) for i in range(0, len(signal) - window_size + 1, window_size)]
    return np.mean(signal)


def signal_detrend(signal):
    """
    Remove linear trend from signal.

    :param signal: Input signal array
    :type signal: np.ndarray
    :return: Detrended signal
    :rtype: np.ndarray
    """
    return detrend(signal, type='linear')


def wavelet_denoising(signal, wavelet='db4', level=1):
    """
    Apply wavelet-based denoising to signal.

    :param signal: Input signal array
    :type signal: np.ndarray
    :param wavelet: Wavelet type
    :type wavelet: str
    :param level: Decomposition level
    :type level: int
    :return: Denoised signal
    :rtype: np.ndarray
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    threshold = np.sqrt(2 * np.log(len(signal))) * np.median(np.abs(coeffs[-level]) / 0.6745)
    new_coeffs = list(map(lambda x: pywt.threshold(x, threshold, mode='soft'), coeffs))
    return pywt.waverec(new_coeffs, wavelet)


def detect_and_interpolate_outliers(data, threshold=3.0):
    """
    Identify and correct statistical outliers in signal.

    :param data: Input signal array
    :type data: np.ndarray
    :param threshold: Standard deviation multiplier threshold
    :type threshold: float
    :return: Corrected signal array
    :rtype: np.ndarray
    """
    # Calculate statistics
    mean = np.mean(data)
    std = np.std(data)

    # Identify outliers
    outliers = np.abs(data - mean) > threshold * std

    if np.any(outliers):
        # Get valid points
        valid_indices = np.where(~outliers)[0]
        valid_data = data[~outliers]

        # Interpolate outliers
        spline = UnivariateSpline(valid_indices, valid_data, k=3, s=0)
        data[outliers] = spline(np.where(outliers)[0])

    return data