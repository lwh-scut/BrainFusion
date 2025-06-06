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


def resample_signal(signal, original_fs, target_fs):
    """
    将信号从原始采样率重采样到目标采样率

    参数:
    signal (numpy array): 原始信号
    original_fs (int): 原始采样率
    target_fs (int): 目标采样率

    返回:
    numpy array: 重采样后的信号
    """
    # 计算降采样因子
    gcd = np.gcd(original_fs, target_fs)  # 计算最大公约数
    up = target_fs // gcd  # 插值因子
    down = original_fs // gcd  # 抽取因子

    # 应用低通滤波器
    nyquist_rate = 0.5 * original_fs
    cutoff = 0.5 * target_fs
    b, a = butter(4, cutoff / nyquist_rate)
    signal_filtered = filtfilt(b, a, signal)

    # 重采样
    signal_resampled = resample_poly(signal_filtered, up, down)

    return signal_resampled


def compute_psd(signal, fs=1000, nperseg=1000, fl=1, fh=40):
    f, Pxx = welch(signal, fs=fs, nperseg=nperseg)
    avg_psd = np.mean(Pxx[(f >= fl) & (f <= fh)])
    return avg_psd


def compute_mean(signal, window_size=None):
    if window_size:
        return [np.mean(signal[i:i + window_size]) for i in range(0, len(signal) - window_size + 1, window_size)]
    else:
        return np.mean(signal)


def signal_detrend(signal):
    """
    Remove linear trend from the signal.

    Parameters:
    signal (numpy.ndarray): Input ECG signal

    Returns:
    numpy.ndarray: Detrended signal
    """
    return detrend(signal, type='linear')


def wavelet_denoising(signal, wavelet='db4', level=1):
    """
    Perform wavelet denoising on the signal.

    Parameters:
    signal (numpy.ndarray): Input ECG signal
    wavelet (str): Type of wavelet to be used for denoising (default 'db4')
    level (int): Decomposition level (default 1)

    Returns:
    numpy.ndarray: Denoised signal
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    threshold = np.sqrt(2 * np.log(len(signal))) * np.median(np.abs(coeffs[-level]) / 0.6745)
    new_coeffs = list(map(lambda x: pywt.threshold(x, threshold, mode='soft'), coeffs))
    return pywt.waverec(new_coeffs, wavelet)


from scipy.interpolate import UnivariateSpline


def detect_and_interpolate_outliers(data, threshold=3.0):
    """
    检测并修复异常点。

    :param data: 输入信号，一维数组
    :param threshold: 异常点检测的阈值，默认为3倍标准差
    :return: 修复后的信号
    """
    # 计算均值和标准差
    mean = np.mean(data)
    std = np.std(data)

    # 检测异常点（大于threshold倍标准差）
    outliers = np.abs(data - mean) > threshold * std

    if np.any(outliers):
        # 找到有效点的索引（非异常点）
        valid_indices = np.where(~outliers)[0]
        valid_data = data[~outliers]

        # 使用样条插值法进行修复
        spline = UnivariateSpline(valid_indices, valid_data, k=3, s=0)
        data[outliers] = spline(np.where(outliers)[0])

    return data
