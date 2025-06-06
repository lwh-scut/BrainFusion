# -*- coding: utf-8 -*-
# @Time    : 2024/3/25 10:37
# @Author  : Li WenHao
# @Site    : South China University of Technology
# @File    : preprocessing.py
# @Software: PyCharm 
# @Comment :
import os.path

import matplotlib.pyplot as plt
import mne
import numpy
import numpy as np
import pywt
import pywt.data
from mne.preprocessing import EOGRegression
from mne_icalabel import label_components
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import iirnotch, medfilt
from BrainFusion.io.File_IO import save_file
from BrainFusion.utils.channels import is_multidimensional_list
from BrainFusion.utils.signals import signal_detrend, wavelet_denoising, detect_and_interpolate_outliers


def wavelet_filter(signal, wavelet='db4', level=4, alpha=1, percent=0.75):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    for i in range(1, len(coeffs)):
        threshold = np.quantile(np.abs(coeffs[i]), percent) * alpha
        coeffs[i] = np.where(np.abs(coeffs[i]) >= threshold, 0, coeffs[i])
    filtered_signal = pywt.waverec(coeffs, wavelet)

    return filtered_signal


# 巴特沃兹滤波器
def butter_bandpass(lowcut, highcut, fs, filter_order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(filter_order, [low, high], btype='band')
    return b, a


# 切比雪夫滤波器
def chebyshev_bandpass(lowcut, highcut, fs, filter_order=4, rp=1):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.cheby1(filter_order, rp, [low, high], btype='band')
    return b, a


# 贝赛尔滤波器
def bessel_bandpass(lowcut, highcut, fs, filter_order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.bessel(filter_order, [low, high], btype='band')
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y


def signal_filter(data, fs, lowcut, highcut, filter_order=4, method='Butterworth'):
    if method == 'Butterworth':
        b, a = butter_bandpass(lowcut, highcut, fs, filter_order)
    elif method == 'Bessel':
        b, a = bessel_bandpass(lowcut, highcut, fs, filter_order)
    elif method == 'chebyshev':
        b, a = chebyshev_bandpass(lowcut, highcut, fs, filter_order, rp=1)
    filter_data = signal.filtfilt(b, a, data, padlen=200)
    return filter_data


# 陷波滤波器
def notch_filter(data, fs, f0=50, Q=30):
    b, a = iirnotch(f0, Q, fs)
    filter_data = signal.filtfilt(b, a, data, padlen=200)
    return filter_data


def multi_notch_filter(data, fs, f0=50, Q=30):
    n = int(int(fs / 2) // f0)
    notch_list = [int(f0 * freq) for freq in range(1, n + 1)]
    filter_data = data.copy()
    for freq in notch_list:
        b, a = iirnotch(freq / fs, Q, fs)
        filter_data = signal.filtfilt(b, a, filter_data, padlen=200)
    return filter_data


def median_filter(data, window_length):
    """
    Apply median filter to each channel of the multi-channel input data.

    Parameters:
    - data (numpy.ndarray): The input multi-channel signal data. Shape should be (n_channels, n_samples).
    - window_length (int): The length of the median filter window. Must be an odd integer.

    Returns:
    - numpy.ndarray: The filtered multi-channel signal.

    Raises:
    - ValueError: If window_length is not an odd integer.
    """
    if window_length % 2 == 0:
        raise ValueError("Each element of window_length should be odd.")

    # Apply median filter to each channel
    filtered_signal = np.apply_along_axis(medfilt, axis=1, arr=data, kernel_size=window_length)

    return filtered_signal


def downsample(data, fs, target_fs):
    factor = int(fs / target_fs)
    downsampled_data = signal.decimate(data, factor)
    return downsampled_data, target_fs


def remove_baseline(data):
    baseline = np.mean(data)
    corrected_data = data - baseline
    return corrected_data


def standardize(data):
    return (data - np.mean(data)) / np.std(data)


def check_channel_list(channel_list, nchan):
    if channel_list:
        return channel_list
    else:
        channel_list = [str(i) for i in range(1, nchan + 1)]
        return channel_list


def eeg_preprocessing(data, chan_list, fs, events, bad_channels, lowcut, highcut, montage='standard_1020',
                      filter_order=4,
                      filter='Butterworth', north_f=50, Q=30,
                      rm_distortion=False, rm_persent=0.05, rm_outlier=False, eog_regression=False, eog_channels=None,
                      is_ICA=False, ICA_component=0, ICA_method='infomax',
                      is_ref=True, ref_chan=None, is_baseline=False, baseline_range=None, is_save=False, save_path=None,
                      save_filestyle='mat '):
    nchan = np.array(data).shape[0]
    length = np.array(data).shape[1]
    filter_data = signal_filter(data=data, fs=fs, lowcut=lowcut, highcut=highcut, filter_order=filter_order,
                                method=filter)
    # 工频陷波
    filter_data = notch_filter(filter_data, f0=north_f, Q=Q, fs=fs)
    # 去除失真段落
    if rm_distortion:
        startIndex = int(length * rm_persent)
        endIndex = int(length - (length * rm_persent))
        filter_data = filter_data[:, startIndex:endIndex]

    # 去除异常点
    def detect_outliers(data):
        # 异常点检测
        mean_val = np.mean(data, axis=1)
        std_val = np.std(data, axis=1)
        threshold = 3 * std_val[:, np.newaxis]
        outliers = np.abs(data - mean_val[:, np.newaxis]) > threshold
        # 标记异常点
        outlier_indices = np.where(outliers)
        # 样条插值
        for channel, time_point in zip(*outlier_indices):
            # 找到非异常点的索引
            non_outlier_indices = np.arange(length)[~outliers[channel, :]]
            # 样条插值
            f = interp1d(non_outlier_indices, data[channel, non_outlier_indices], kind='cubic',
                         fill_value="extrapolate")
            # 用插值填充异常点
            data[channel, time_point] = f(time_point)

    if rm_outlier:
        detect_outliers(filter_data)
    info = mne.create_info(ch_names=chan_list, sfreq=fs, ch_types='eeg')
    raw = mne.io.RawArray(data=filter_data, info=info)

    if eog_regression:
        if eog_channels:
            raw.set_eeg_reference('average', projection=False)
            raw.set_channel_types({ch: 'eog' for ch in eog_channels})
            # Fit the regression model
            weights = EOGRegression().fit(raw)
            raw = weights.apply(raw, copy=True)
    try:
        raw.set_montage(montage)
    except Exception:
        print("montage is none or not standard")
    if len(bad_channels[0]) != 0:
        raw.drop_channels(bad_channels)
    # 进行ICA去伪迹
    try:
        if is_ICA:
            print("ICA")
            ica = mne.preprocessing.ICA(n_components=ICA_component, method=ICA_method)
            ica.fit(raw)
            ic_labels = label_components(raw, ica, method='iclabel')
            if eog_regression:
                exclude_idx = [idx for idx, label in enumerate(ic_labels['labels']) if
                               label not in ['brain', 'other', 'eye blink']]
            else:
                exclude_idx = [idx for idx, label in enumerate(ic_labels['labels']) if label not in ['brain', 'other']]
            print(f"Excluding these ICA components:{exclude_idx}")
            ica.apply(raw, exclude=exclude_idx)
    except Exception as e:
        print(e)
    if not eog_regression and is_ref:
        if len(ref_chan[0]) != 0:
            raw.set_eeg_reference(ref_chan, projection=False)
        else:
            raw.set_eeg_reference('average', projection=False)

    baseline_correction_data = raw.get_data()
    # 基线校正
    if is_baseline:
        if baseline_range:
            baseline_data = raw.get_data()[:, baseline_range[0]:baseline_range[1]]
            baseline_correction_data = raw.get_data() - np.mean(baseline_data, axis=1, keepdims=True)

    preprocessing_data = {}
    preprocessing_data['data'] = baseline_correction_data
    preprocessing_data['srate'] = fs
    preprocessing_data['nchan'] = len(raw.ch_names)
    preprocessing_data['ch_names'] = raw.ch_names
    preprocessing_data['events'] = events
    preprocessing_data['type'] = 'eeg_preprocess'
    preprocessing_data['montage'] = montage
    # 数据保存
    if is_save:
        save_file(data=preprocessing_data, save_path=save_path, save_filestyle=save_filestyle)
    return preprocessing_data


def eeg_preprocessing_by_dict(data_dict, lowcut, highcut, bad_channels, montage='standard_1020', notch_f=50,
                      rm_outlier=False, eog_regression=False, eog_channels=None,
                      is_ICA=False, ICA_component=0, ICA_method='infomax',
                      is_ref=True, ref_chan=None, is_save=False, save_path=None,
                      save_filestyle='mat'):
    data = data_dict['data']
    fs = data_dict['srate']
    chan_list = list(data_dict['ch_names'])
    length = np.array(data).shape[1]
    info = mne.create_info(ch_names=chan_list, sfreq=fs, ch_types='eeg')
    raw = mne.io.RawArray(data=data, info=info)
    try:
        raw.set_montage(montage)
    except Exception:
        print("montage is none or not standard")

        # 1. 滤波（带通滤波）
    raw.filter(l_freq=lowcut, h_freq=highcut, fir_design='firwin', phase='zero', verbose=True)

    # 2. 工频陷波（Notch Filter）
    notch_freqs = np.arange(notch_f, fs / 2, notch_f)  # 默认工频为 50Hz，阶乘为 notch_f 的倍数
    raw.notch_filter(freqs=notch_freqs, notch_widths=2, verbose=True)

    if eog_regression:
        if eog_channels:
            raw.set_eeg_reference('average', projection=False)
            raw.set_channel_types({ch: 'eog' for ch in eog_channels})
            # Fit the regression model
            weights = EOGRegression().fit(raw)
            raw = weights.apply(raw, copy=True)

    if len(bad_channels[0]) != 0:
        raw.drop_channels(bad_channels)

    # 3. 样条插值
    if rm_outlier:
        for i in range(len(chan_list)):
            raw._data[i] = detect_and_interpolate_outliers(raw._data[i])
    # 进行ICA去伪迹
    try:
        if is_ICA:
            print("ICA")
            ica = mne.preprocessing.ICA(n_components=ICA_component, method=ICA_method)
            ica.fit(raw)
            ic_labels = label_components(raw, ica, method='iclabel')
            if eog_regression:
                exclude_idx = [idx for idx, label in enumerate(ic_labels['labels']) if
                               label not in ['brain', 'other']]
            else:
                exclude_idx = [idx for idx, label in enumerate(ic_labels['labels']) if label not in ['brain', 'other']]
            print(f"Excluding these ICA components:{exclude_idx}")
            ica.apply(raw, exclude=exclude_idx)
    except Exception as e:
        print(e)
    if not eog_regression and is_ref:
        if len(ref_chan[0]) != 0:
            raw.set_eeg_reference(ref_chan, projection=False)
        else:
            raw.set_eeg_reference('average', projection=False)

    baseline_correction_data = raw.get_data()
    preprocessing_data = data_dict.copy()
    preprocessing_data['data'] = baseline_correction_data
    preprocessing_data['srate'] = fs
    preprocessing_data['nchan'] = len(raw.ch_names)
    preprocessing_data['ch_names'] = raw.ch_names
    preprocessing_data['type'] = 'eeg_preprocess'
    preprocessing_data['montage'] = montage

    # 数据保存
    if is_save:
        save_file(data=preprocessing_data, save_path=save_path, save_filestyle=save_filestyle)
    return preprocessing_data


def fnirs_preprocessing(data, channel_list, fs, light_freqs, events, src_pos, det_pos, lowcut=0.01, highcut=0.7,
                        filter_order=3,
                        wavelet='db1', level=4, alpha=1, percent=0.75, is_save=False, save_path=None,
                        save_filestyle='mat'):
    def trans_list_to_str(channel_list):
        hbo_list = []
        hbr_list = []
        nchan = len(channel_list)
        for i in range(0, int(nchan / 2)):
            ch_name = str(+[i][0]) + '_' + str(channel_list[i][1]) + '_' + 'hbo'
            hbo_list.append(ch_name)
        for i in range(int(nchan / 2), nchan):
            ch_name = str(channel_list[i][0]) + '_' + str(channel_list[i][1]) + '_' + 'hbr'
            hbr_list.append(ch_name)
        hbo_list.extend(hbr_list)
        return hbo_list

    def trans_str_to_list(channel_name):
        channel_list = []
        for ch_name in channel_name:
            sp_name = ch_name.split('_')
            channel_list.append([int(sp_name[0]), int(sp_name[1])])
        return channel_list

    od_data = fnirs_trans_optical_density(data)
    denose_od_data = [wavelet_filter(od, wavelet=wavelet, level=level, alpha=alpha, percent=percent) for od in od_data]
    b, a = bessel_bandpass(lowcut=lowcut, highcut=highcut, fs=fs, filter_order=filter_order)
    filter_data = signal.filtfilt(b, a, denose_od_data)
    nchan = len(channel_list)
    channel_list = trans_str_to_list(channel_list)
    hbo, hbr = fnirs_beer_lambert_law(data=filter_data, nchan=nchan, freqs=light_freqs, channel_list=channel_list,
                                      src_pos=src_pos, det_pos=det_pos)
    hbo = hbo.tolist()
    hbr = hbr.tolist()
    hbo.extend(hbr)

    preprocessing_data = {}
    preprocessing_data['data'] = hbo
    preprocessing_data['freqs'] = light_freqs
    preprocessing_data['srate'] = fs
    preprocessing_data['nchan'] = nchan
    preprocessing_data['ch_names'] = trans_list_to_str(channel_list)
    preprocessing_data['events'] = events
    preprocessing_data['type'] = 'fnirs_preprocess'
    preprocessing_data['montage'] = None
    preprocessing_data['loc'] = [src_pos, det_pos]

    if is_save:
        save_file(data=preprocessing_data, save_path=save_path, save_filestyle=save_filestyle)
    return preprocessing_data


def fnirs_preprocessing_by_raw(raw, bad_channels, lowcut=0.01, highcut=0.7, wavelet='db4', level=4, alpha=0.2, enable_interpolate=False):
    # 复制原始数据
    raw_intensity = raw.copy()

    if len(bad_channels[0]) != 0:
        raw_intensity.drop_channels(bad_channels)
    # 步骤 1: 光学密度转换
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
    # 步骤 2: 计算HbO和HbR浓度
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)
    # 步骤 3: 带通滤波（0.01到0.2 Hz）
    filtered_data = mne.filter.filter_data(raw_haemo.get_data(), raw_haemo.info['sfreq'],
                                          lowcut, highcut, filter_length='auto',
                                          method='fir', iir_params=None)
    raw_haemo._data = filtered_data
    # 步骤 4: 运动伪影去除（使用小波去除运动伪影）
    if wavelet:
        for ch in range(len(raw_haemo.info['ch_names'])):
            raw_haemo._data[ch] = wavelet_denoising(raw_haemo._data[ch], wavelet, level, alpha)
    # 步骤 5: 异常点检测与修复
    if enable_interpolate:
        for ch in range(len(raw_haemo.info['ch_names'])):
            raw_haemo._data[ch] = detect_and_interpolate_outliers(raw_haemo._data[ch])

    # 步骤 6: 去趋势
    raw_haemo._data = signal.detrend(raw_haemo.get_data(), axis=1)

    # 步骤 7: 标准化 (z-scoring)
    raw_haemo._data = (raw_haemo.get_data() - np.mean(raw_haemo.get_data(), axis=1, keepdims=True)) / np.std(raw_haemo.get_data(), axis=1, keepdims=True)

    return raw_haemo


def wavelet_denoising(data, wavelet='db1', level=4, alpha=1):
    """
    小波去噪函数
    :param data: 输入数据，应该是一个一维数组
    :param wavelet: 使用的小波类型
    :param level: 小波分解层数
    :param alpha: 去噪参数
    :return: 去噪后的数据
    """
    # 小波分解
    coeffs = pywt.wavedec(data, wavelet, level=level)

    # 通过阈值去噪
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], alpha * np.max(coeffs[i]), mode='soft')

    # 小波重构
    denoised_data = pywt.waverec(coeffs, wavelet)

    # 确保去噪后的数据长度与原数据一致
    if len(denoised_data) > len(data):
        denoised_data = denoised_data[:len(data)]  # 裁剪多余部分
    elif len(denoised_data) < len(data):
        denoised_data = np.pad(denoised_data, (0, len(data) - len(denoised_data)), 'constant')  # 填充不足部分

    return denoised_data



def fnirs_trans_optical_density(data):
    od_list = []
    for signal in data:
        data_mean = np.mean(signal)
        signal /= data_mean
        signal = np.log(signal)
        signal *= -1
        od_list.append(signal)
    return od_list


def fnirs_beer_lambert_law(data, nchan, freqs, channel_list, src_pos, det_pos, ppf=6.):
    def cal_distance(chan_list, src_pos, det_pos):
        distance = []
        for chan in chan_list:
            src_index = chan[0] - 1
            det_index = chan[1] - 1
            dis = np.linalg.norm(np.diff((src_pos[src_index], det_pos[det_index])))
            distance.append(dis)
        return distance

    def load_absorption(freqs):
        from scipy.io import loadmat
        from scipy.interpolate import interp1d
        extinction_fname = os.path.join(os.path.dirname(__file__), 'extinction_coef.mat')
        a = loadmat(extinction_fname)['extinct_coef']
        interp_hbo = interp1d(a[:, 0], a[:, 1], kind='linear')
        interp_hb = interp1d(a[:, 0], a[:, 2], kind='linear')
        ext_coef = np.array([[interp_hbo(freqs[0]), interp_hb(freqs[0])],
                             [interp_hbo(freqs[1]), interp_hb(freqs[1])]])
        abs_coef = ext_coef * 0.2303
        return abs_coef

    abs_coef = load_absorption(freqs)
    distances = cal_distance(channel_list, src_pos, det_pos)
    hbo_list = []
    hbr_list = []
    for ii, jj in zip(range(0, int(nchan / 2)), range(int(nchan / 2), nchan)):
        EL = abs_coef * distances[ii] * ppf
        iEL = np.linalg.pinv(EL)
        hbf = iEL @ data[[ii, jj]] * 1e-3
        hbo_list.append(hbf[0])
        hbr_list.append(hbf[1])
    return np.array(hbo_list), np.array(hbr_list)


def ecg_preprocessing(ecg_signal, fs, lowcut, highcut, events=None, chan_list=None, downsample_fs=None,
                      filter_order=4, filter='Butterworth', north_f=50, Q=30, is_rm_baseline=True,
                      median_filter_window=0.2, is_wavelet_denoise=False, wavelet='db4', level=1,
                      is_save=False, save_path=None, save_filestyle='mat'):
    # Bandpass filter
    print(ecg_signal)
    filtered_signal = signal_filter(ecg_signal, fs=fs, lowcut=lowcut, highcut=highcut, filter_order=filter_order,
                                    method=filter)
    print(filtered_signal)
    # Notch filter
    filtered_signal = notch_filter(data=filtered_signal, fs=fs, f0=north_f, Q=Q)

    # Downsample
    if downsample_fs:
        filtered_signal, fs = downsample(filtered_signal, fs, downsample_fs)

    # Detrend
    detrended_signal = signal_detrend(filtered_signal)
    print(detrended_signal)

    # Wavelet denoising (optional)
    if is_wavelet_denoise:
        denoised_signal = wavelet_denoising(detrended_signal, wavelet=wavelet, level=level)
    else:
        denoised_signal = detrended_signal

    # Median filter for noise removal
    window_length = int(median_filter_window * fs)  # 200ms window
    if window_length:
        if window_length % 2 == 0:  # Ensure the window length is odd
            window_length += 1
        median_filtered_signal = median_filter(denoised_signal, window_length)
    else:
        median_filtered_signal = denoised_signal

    # Baseline correction
    baseline_corrected_signal = remove_baseline(median_filtered_signal)

    # Data standardization
    standardized_signal = (baseline_corrected_signal - np.mean(baseline_corrected_signal)) / np.std(
        baseline_corrected_signal)

    if isinstance(ecg_signal, list):
        nchan = len(ecg_signal)
    elif isinstance(ecg_signal, numpy.ndarray):
        nchan = ecg_signal.shape[0]
    else:
        raise ValueError('Data type error.Data must be list or numpy.ndarray')
    chan_list = check_channel_list(chan_list, nchan)

    preprocessed_data = {}
    preprocessed_data['data'] = standardized_signal
    preprocessed_data['srate'] = fs
    preprocessed_data['nchan'] = nchan
    preprocessed_data['ch_names'] = chan_list
    preprocessed_data['events'] = events
    preprocessed_data['type'] = 'ecg_preprocess'
    preprocessed_data['montage'] = None

    # save preprocessed data
    if is_save:
        save_file(data=preprocessed_data, save_path=save_path, save_filestyle=save_filestyle)
    return preprocessed_data


def emg_preprocessing(emg_signal, fs, bf_lowcut, bf_highcut, lf_cutoff, events=None, chan_list=None,
                      bf_order=6, filter='Butterworth', lf_order=4, north_f=50, Q=30,
                      is_save=False, save_path=None, save_filestyle='mat'):
    # Bandpass filter
    filtered_signal = signal_filter(emg_signal, fs=fs, lowcut=bf_lowcut, highcut=bf_highcut, filter_order=bf_order,
                                    method=filter)

    # Notch filter
    filtered_signal = multi_notch_filter(data=filtered_signal, fs=fs, f0=north_f, Q=Q)

    # rectify
    rectify_signal = np.abs(filtered_signal)

    # Lowpass_filter
    lowpass_signal = butter_lowpass_filter(data=rectify_signal, fs=fs, cutoff=lf_cutoff, order=lf_order)

    # Data standardization
    standardized_signal = (lowpass_signal - np.mean(lowpass_signal)) / np.std(lowpass_signal)

    if isinstance(emg_signal, list):
        nchan = len(emg_signal)
        if is_multidimensional_list(emg_signal):
            emg_signal = np.array(emg_signal).reshape(1, -1)
    elif isinstance(emg_signal, numpy.ndarray):
        if emg_signal.ndim == 1:
            emg_signal = np.array(emg_signal).reshape(1, -1)
        nchan = emg_signal.shape[0]
    else:
        raise ValueError('Data type error.Data must be list or numpy.ndarray')
    chan_list = check_channel_list(chan_list, nchan)
    preprocessed_data = {}
    preprocessed_data['data'] = standardized_signal
    preprocessed_data['srate'] = fs
    preprocessed_data['nchan'] = nchan
    preprocessed_data['ch_names'] = chan_list
    preprocessed_data['events'] = events
    preprocessed_data['type'] = 'emg_preprocess'
    preprocessed_data['montage'] = None

    # save preprocessed data
    if is_save:
        save_file(data=preprocessed_data, save_path=save_path, save_filestyle=save_filestyle)
    return preprocessed_data


def create_epoch(data, method, is_save, save_path, save_filestyle='mat', events_list=None, events_range=None,
                 fix_length=None, custom_events=None, is_point=False):
    epoch_data = []
    info = data.copy()
    info['data'] = None
    array_data = np.array(data['data'])
    if is_point:
        srate = 1
    else:
        srate = data['srate']
    if data:
        if method == 'split_by_events':
            if events_list:
                if len(events_list) > 1:
                    for i in range(len(events_list) - 1):
                        start = int(events_list[i] * srate)
                        end = int(events_list[i + 1] * srate)
                        print(start, end)
                        epoch_data.append(array_data[:, start:end])

        elif method == 'split_by_front_and_back_of_events':
            if events_list and events_range:
                for event in events_list:
                    event = int(event * data['srate'])
                    epoch_data.append(array_data[:, event - events_range[0]:event + events_range[1]])

        elif method == 'split_by_fixed_length':
            if fix_length:
                num_epoch = np.array(data['data']).shape[1] // fix_length
                for i in range(num_epoch):
                    epoch_data.append(array_data[:, i * fix_length:i * fix_length + fix_length])

        elif method == 'custom_split':
            pass

        if is_save:
            # 检测文件夹是否存在
            if not os.path.exists(save_path):
                # 如果不存在，则创建文件夹
                os.makedirs(save_path)
            for i in range(len(epoch_data)):
                epoch = info.copy()
                epoch['data'] = epoch_data[i]
                epoch['events'] = None
                if is_save:
                    save_file(data=epoch, save_path=os.path.join(save_path, str(i + 1) + '.' + save_filestyle),
                              save_filestyle=save_filestyle)
