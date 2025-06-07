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
import scipy
from mne.preprocessing import EOGRegression
from mne_icalabel import label_components
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import iirnotch, medfilt
from BrainFusion.io.File_IO import save_file
from BrainFusion.utils.channels import is_multidimensional_list
from BrainFusion.utils.signals import signal_detrend, wavelet_denoising, detect_and_interpolate_outliers

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
    """
    Apply wavelet filtering to remove noise from signals.

    :param signal: Input signal data
    :type signal: numpy.ndarray
    :param wavelet: Wavelet type (default: 'db4')
    :type wavelet: str
    :param level: Decomposition level
    :type level: int
    :param alpha: Threshold multiplier
    :type alpha: float
    :param percent: Coefficient retention percentile
    :type percent: float
    :return: Filtered signal
    :rtype: numpy.ndarray
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    for i in range(1, len(coeffs)):
        threshold = np.quantile(np.abs(coeffs[i]), percent) * alpha
        coeffs[i] = np.where(np.abs(coeffs[i]) >= threshold, 0, coeffs[i])
    filtered_signal = pywt.waverec(coeffs, wavelet)
    return filtered_signal


def butter_bandpass(lowcut, highcut, fs, filter_order=4):
    """
    Design Butterworth bandpass filter coefficients.

    :param lowcut: Low cutoff frequency (Hz)
    :type lowcut: float
    :param highcut: High cutoff frequency (Hz)
    :type highcut: float
    :param fs: Sampling frequency (Hz)
    :type fs: float
    :param filter_order: Filter order (default: 4)
    :type filter_order: int
    :return: Numerator (b) and denominator (a) polynomials
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(filter_order, [low, high], btype='band')
    return b, a


def chebyshev_bandpass(lowcut, highcut, fs, filter_order=4, rp=1):
    """
    Design Chebyshev Type I bandpass filter coefficients.

    :param lowcut: Low cutoff frequency (Hz)
    :type lowcut: float
    :param highcut: High cutoff frequency (Hz)
    :type highcut: float
    :param fs: Sampling frequency (Hz)
    :type fs: float
    :param filter_order: Filter order (default: 4)
    :type filter_order: int
    :param rp: Peak-to-peak ripple (dB)
    :type rp: float
    :return: Numerator (b) and denominator (a) polynomials
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.cheby1(filter_order, rp, [low, high], btype='band')
    return b, a


def bessel_bandpass(lowcut, highcut, fs, filter_order=4):
    """
    Design Bessel bandpass filter coefficients.

    :param lowcut: Low cutoff frequency (Hz)
    :type lowcut: float
    :param highcut: High cutoff frequency (Hz)
    :type highcut: float
    :param fs: Sampling frequency (Hz)
    :type fs: float
    :param filter_order: Filter order (default: 4)
    :type filter_order: int
    :return: Numerator (b) and denominator (a) polynomials
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.bessel(filter_order, [low, high], btype='band')
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=4):
    """
    Apply Butterworth lowpass filter to signal data.

    :param data: Input signal data
    :type data: numpy.ndarray
    :param cutoff: Cutoff frequency (Hz)
    :type cutoff: float
    :param fs: Sampling frequency (Hz)
    :type fs: float
    :param order: Filter order (default: 4)
    :type order: int
    :return: Filtered signal data
    :rtype: numpy.ndarray
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y


def signal_filter(data, fs, lowcut, highcut, filter_order=4, method='Butterworth'):
    """
    Apply bandpass filter to signal data.

    :param data: Input signal data
    :type data: numpy.ndarray
    :param fs: Sampling frequency (Hz)
    :type fs: float
    :param lowcut: Low cutoff frequency (Hz)
    :type lowcut: float
    :param highcut: High cutoff frequency (Hz)
    :type highcut: float
    :param filter_order: Filter order (default: 4)
    :type filter_order: int
    :param method: Filter design method (default: 'Butterworth')
    :type method: str
    :return: Filtered signal data
    :rtype: numpy.ndarray
    """
    if method == 'Butterworth':
        b, a = butter_bandpass(lowcut, highcut, fs, filter_order)
    elif method == 'Bessel':
        b, a = bessel_bandpass(lowcut, highcut, fs, filter_order)
    elif method == 'chebyshev':
        b, a = chebyshev_bandpass(lowcut, highcut, fs, filter_order, rp=1)
    filter_data = signal.filtfilt(b, a, data, padlen=200)
    return filter_data


def notch_filter(data, fs, f0=50, Q=30):
    """
    Apply notch filter to remove powerline interference.

    :param data: Input signal data
    :type data: numpy.ndarray
    :param fs: Sampling frequency (Hz)
    :type fs: float
    :param f0: Notch frequency (default: 50 Hz)
    :type f0: float
    :param Q: Quality factor (default: 30)
    :type Q: float
    :return: Filtered signal data
    :rtype: numpy.ndarray
    """
    b, a = iirnotch(f0, Q, fs)
    filter_data = signal.filtfilt(b, a, data, padlen=200)
    return filter_data


def multi_notch_filter(data, fs, f0=50, Q=30):
    """
    Apply multiple notch filters to remove harmonics.

    :param data: Input signal data
    :type data: numpy.ndarray
    :param fs: Sampling frequency (Hz)
    :type fs: float
    :param f0: Fundamental frequency (default: 50 Hz)
    :type f0: float
    :param Q: Quality factor (default: 30)
    :type Q: float
    :return: Filtered signal data
    :rtype: numpy.ndarray
    """
    n = int(int(fs / 2) // f0)
    notch_list = [int(f0 * freq) for freq in range(1, n + 1)]
    filter_data = data.copy()
    for freq in notch_list:
        b, a = iirnotch(freq, Q, fs)
        filter_data = signal.filtfilt(b, a, filter_data, padlen=200)
    return filter_data


def median_filter(data, window_length):
    """
    Apply median filtering to signal data.

    :param data: Input signal data (n_channels × n_samples)
    :type data: numpy.ndarray
    :param window_length: Median filter window length (must be odd)
    :type window_length: int
    :return: Filtered signal data
    :rtype: numpy.ndarray
    :raises ValueError: If window_length is even
    """
    if window_length % 2 == 0:
        raise ValueError("Window length must be odd.")
    filtered_signal = np.apply_along_axis(medfilt, axis=1, arr=data, kernel_size=window_length)
    return filtered_signal


def downsample(data, fs, target_fs):
    """
    Downsample signal data to target sampling rate.

    :param data: Input signal data
    :type data: numpy.ndarray
    :param fs: Original sampling frequency (Hz)
    :type fs: float
    :param target_fs: Target sampling frequency (Hz)
    :type target_fs: float
    :return: Downsampled data and new sampling rate
    :rtype: tuple(numpy.ndarray, float)
    """
    factor = int(fs / target_fs)
    downsampled_data = signal.decimate(data, factor)
    return downsampled_data, target_fs


def remove_baseline(data):
    """
    Remove DC offset from signal data.

    :param data: Input signal data
    :type data: numpy.ndarray
    :return: Baseline-corrected data
    :rtype: numpy.ndarray
    """
    baseline = np.mean(data)
    corrected_data = data - baseline
    return corrected_data


def standardize(data):
    """
    Standardize signal data to zero mean and unit variance.

    :param data: Input signal data
    :type data: numpy.ndarray
    :return: Standardized data
    :rtype: numpy.ndarray
    """
    return (data - np.mean(data)) / np.std(data)


def check_channel_list(channel_list, nchan):
    """
    Generate channel list if none provided.

    :param channel_list: Existing channel names or None
    :type channel_list: list or None
    :param nchan: Number of channels
    :type nchan: int
    :return: Channel names list
    :rtype: list
    """
    if channel_list:
        return channel_list
    else:
        return [str(i) for i in range(1, nchan + 1)]


def eeg_preprocessing(data, chan_list, fs, events, bad_channels, lowcut, highcut, montage='standard_1020',
                      filter_order=4, filter_method='Butterworth', notch_f=50, Q=30,
                      rm_distortion=False, rm_percent=0.05, rm_outlier=False, eog_regression=False, eog_channels=None,
                      is_ICA=False, ICA_component=0, ICA_method='infomax',
                      is_ref=True, ref_chan=None, is_baseline=False, baseline_range=None,
                      is_save=False, save_path=None, save_filestyle='mat'):
    """
    Comprehensive EEG preprocessing pipeline with artifact removal options.

    :param data: Raw EEG data (n_channels × n_samples)
    :type data: numpy.ndarray
    :param chan_list: Channel names list
    :type chan_list: list
    :param fs: Sampling frequency (Hz)
    :type fs: float
    :param events: Event markers
    :type events: list
    :param bad_channels: Channels to exclude
    :type bad_channels: list
    :param lowcut: Bandpass low cutoff frequency (Hz)
    :type lowcut: float
    :param highcut: Bandpass high cutoff frequency (Hz)
    :type highcut: float
    :param montage: Electrode montage (default: 'standard_1020')
    :type montage: str
    :param filter_order: Bandpass filter order (default: 4)
    :type filter_order: int
    :param filter_method: Filter design method (default: 'Butterworth')
    :type filter_method: str
    :param notch_f: Notch frequency (default: 50 Hz)
    :type notch_f: float
    :param Q: Notch filter quality factor (default: 30)
    :type Q: float
    :param rm_distortion: Trim signal edges (default: False)
    :type rm_distortion: bool
    :param rm_percent: Edge trim percentage (default: 0.05)
    :type rm_percent: float
    :param rm_outlier: Detect and interpolate outliers (default: False)
    :type rm_outlier: bool
    :param eog_regression: Apply EOG regression (default: False)
    :type eog_regression: bool
    :param eog_channels: EOG channel names (default: None)
    :type eog_channels: list or None
    :param is_ICA: Apply ICA artifact removal (default: False)
    :type is_ICA: bool
    :param ICA_component: Number of ICA components (0=auto)
    :type ICA_component: int
    :param ICA_method: ICA algorithm (default: 'infomax')
    :type ICA_method: str
    :param is_ref: Apply rereferencing (default: True)
    :type is_ref: bool
    :param ref_chan: Reference channel(s) (default: None=average)
    :type ref_chan: list or None
    :param is_baseline: Apply baseline correction (default: False)
    :type is_baseline: bool
    :param baseline_range: Baseline time window [start,end] (samples)
    :type baseline_range: list or None
    :param is_save: Save preprocessed data (default: False)
    :type is_save: bool
    :param save_path: Output file path
    :type save_path: str
    :param save_filestyle: Output file format
    :type save_filestyle: str
    :return: Preprocessed EEG data structure
    :rtype: dict
    """
    nchan, length = data.shape

    # Apply bandpass filtering
    filtered_data = signal_filter(
        data=data,
        fs=fs,
        lowcut=lowcut,
        highcut=highcut,
        filter_order=filter_order,
        method=filter_method
    )

    # Apply notch filtering
    filtered_data = notch_filter(filtered_data, fs=fs, f0=notch_f, Q=Q)

    # Trim beginning and end segments
    if rm_distortion:
        start = int(length * rm_percent)
        end = int(length - (length * rm_percent))
        filtered_data = filtered_data[:, start:end]
        length = end - start  # Update signal length

    # Create MNE Raw object
    info = mne.create_info(ch_names=chan_list, sfreq=fs, ch_types='eeg')
    raw = mne.io.RawArray(data=filtered_data, info=info)

    # Apply EOG regression
    if eog_regression and eog_channels:
        raw.set_eeg_reference('average', projection=False)
        raw.set_channel_types({ch: 'eog' for ch in eog_channels})
        weights = EOGRegression().fit(raw)
        raw = weights.apply(raw, copy=True)

    # Set electrode positions
    try:
        raw.set_montage(montage)
    except Exception:
        print("Non-standard or missing montage")

    # Remove bad channels
    if bad_channels:
        raw.drop_channels(bad_channels)

    # Apply ICA artifact removal
    if is_ICA:
        ica = mne.preprocessing.ICA(
            n_components=ICA_component,
            method=ICA_method
        )
        ica.fit(raw)
        ic_labels = label_components(raw, ica, method='iclabel')

        # Determine components to exclude
        if eog_regression:
            exclude_idx = [idx for idx, label in enumerate(ic_labels['labels'])
                           if label not in ['brain', 'other']]
        else:
            exclude_idx = [idx for idx, label in enumerate(ic_labels['labels'])
                           if label not in ['brain', 'other']]
        print(f"Excluded ICA components: {exclude_idx}")
        ica.apply(raw, exclude=exclude_idx)

    # Apply rereferencing
    if not eog_regression and is_ref:
        if ref_chan:
            raw.set_eeg_reference(ref_chan, projection=False)
        else:
            raw.set_eeg_reference('average', projection=False)

    # Get processed data
    processed_data = raw.get_data()

    # Apply baseline correction
    if is_baseline and baseline_range:
        baseline = processed_data[:, baseline_range[0]:baseline_range[1]]
        processed_data = processed_data - np.mean(baseline, axis=1, keepdims=True)

    # Prepare output structure
    result = {
        'data': processed_data,
        'srate': fs,
        'nchan': len(raw.ch_names),
        'ch_names': raw.ch_names,
        'events': events,
        'type': 'eeg_preprocess',
        'montage': montage
    }

    # Save results
    if is_save:
        save_file(data=result, save_path=save_path, save_filestyle=save_filestyle)

    return result


def eeg_preprocessing_by_dict(data_dict, lowcut, highcut, bad_channels, montage='standard_1020', notch_f=50,
                              rm_outlier=False, eog_regression=False, eog_channels=None,
                              is_ICA=False, ICA_component=0, ICA_method='infomax',
                              is_ref=True, ref_chan=None, is_save=False, save_path=None,
                              save_filestyle='mat'):
    """
    EEG preprocessing pipeline using dictionary input/output format.

    :param data_dict: Raw EEG data structure
    :type data_dict: dict
    :param lowcut: Bandpass low cutoff frequency (Hz)
    :type lowcut: float
    :param highcut: Bandpass high cutoff frequency (Hz)
    :type highcut: float
    :param bad_channels: Channels to exclude
    :type bad_channels: list
    :param montage: Electrode montage (default: 'standard_1020')
    :type montage: str
    :param notch_f: Notch frequency (default: 50 Hz)
    :type notch_f: float
    :param rm_outlier: Detect and interpolate outliers (default: False)
    :type rm_outlier: bool
    :param eog_regression: Apply EOG regression (default: False)
    :type eog_regression: bool
    :param eog_channels: EOG channel names (default: None)
    :type eog_channels: list or None
    :param is_ICA: Apply ICA artifact removal (default: False)
    :type is_ICA: bool
    :param ICA_component: Number of ICA components (0=auto)
    :type ICA_component: int
    :param ICA_method: ICA algorithm (default: 'infomax')
    :type ICA_method: str
    :param is_ref: Apply rereferencing (default: True)
    :type is_ref: bool
    :param ref_chan: Reference channel(s) (default: None=average)
    :type ref_chan: list or None
    :param is_save: Save preprocessed data (default: False)
    :type is_save: bool
    :param save_path: Output file path
    :type save_path: str
    :param save_filestyle: Output file format
    :type save_filestyle: str
    :return: Preprocessed EEG data structure
    :rtype: dict
    """
    # Extract parameters from input dictionary
    data = data_dict['data']
    fs = data_dict['srate']
    chan_list = list(data_dict['ch_names'])

    # Create MNE Raw object
    info = mne.create_info(ch_names=chan_list, sfreq=fs, ch_types='eeg')
    raw = mne.io.RawArray(data=data, info=info)

    # Set electrode positions
    try:
        raw.set_montage(montage)
    except Exception:
        print("Non-standard or missing montage")

    # Apply bandpass filtering
    raw.filter(
        l_freq=lowcut,
        h_freq=highcut,
        fir_design='firwin',
        phase='zero',
        verbose=True
    )

    # Apply harmonic notch filtering
    notch_freqs = np.arange(notch_f, fs / 2, notch_f)
    raw.notch_filter(freqs=notch_freqs, notch_widths=2, verbose=True)

    # Apply EOG regression
    if eog_regression and eog_channels:
        raw.set_eeg_reference('average', projection=False)
        raw.set_channel_types({ch: 'eog' for ch in eog_channels})
        weights = EOGRegression().fit(raw)
        raw = weights.apply(raw, copy=True)

    # Remove bad channels
    if bad_channels:
        raw.drop_channels(bad_channels)

    # Detect and interpolate outliers
    if rm_outlier:
        for i in range(len(chan_list)):
            raw._data[i] = detect_and_interpolate_outliers(raw._data[i])

    # Apply ICA artifact removal
    if is_ICA:
        ica = mne.preprocessing.ICA(
            n_components=ICA_component,
            method=ICA_method
        )
        ica.fit(raw)
        ic_labels = label_components(raw, ica, method='iclabel')

        # Determine components to exclude
        exclude_idx = [idx for idx, label in enumerate(ic_labels['labels'])
                       if label not in ['brain', 'other']]
        print(f"Excluded ICA components: {exclude_idx}")
        ica.apply(raw, exclude=exclude_idx)

    # Apply rereferencing
    if not eog_regression and is_ref:
        if ref_chan:
            raw.set_eeg_reference(ref_chan, projection=False)
        else:
            raw.set_eeg_reference('average', projection=False)

    # Prepare output structure
    result = data_dict.copy()
    result.update({
        'data': raw.get_data(),
        'nchan': len(raw.ch_names),
        'ch_names': raw.ch_names,
        'type': 'eeg_preprocess',
        'montage': montage
    })

    # Save results
    if is_save:
        save_file(data=result, save_path=save_path, save_filestyle=save_filestyle)

    return result


def fnirs_preprocessing(data, channel_list, fs, light_freqs, events, src_pos, det_pos, lowcut=0.01, highcut=0.7,
                        filter_order=3, wavelet='db1', level=4, alpha=1, percent=0.75,
                        is_save=False, save_path=None, save_filestyle='mat'):
    """
    Preprocess fNIRS data including optical density conversion, wavelet filtering, and haemoglobin calculation.

    :param data: Raw optical intensity data
    :type data: list or numpy.ndarray
    :param channel_list: Source-detector pairing information
    :type channel_list: list
    :param fs: Sampling frequency (Hz)
    :type fs: float
    :param light_freqs: Light wavelength frequencies [λ1, λ2] (nm)
    :type light_freqs: list
    :param events: Event markers information
    :type events: list
    :param src_pos: Source positions coordinates
    :type src_pos: list
    :param det_pos: Detector positions coordinates
    :type det_pos: list
    :param lowcut: Low cutoff frequency for bandpass filter (Hz, default: 0.01)
    :type lowcut: float
    :param highcut: High cutoff frequency for bandpass filter (Hz, default: 0.7)
    :type highcut: float
    :param filter_order: Bandpass filter order (default: 3)
    :type filter_order: int
    :param wavelet: Wavelet type for denoising (default: 'db1')
    :type wavelet: str
    :param level: Wavelet decomposition level (default: 4)
    :type level: int
    :param alpha: Wavelet threshold multiplier (default: 1)
    :type alpha: float
    :param percent: Wavelet coefficient retention percentile (default: 0.75)
    :type percent: float
    :param is_save: Save preprocessing results flag (default: False)
    :type is_save: bool
    :param save_path: Output file save path
    :type save_path: str
    :param save_filestyle: Output file format (default: 'mat')
    :type save_filestyle: str
    :return: Preprocessed fNIRS data structure
    :rtype: dict
    """

    def trans_list_to_str(ch_list):
        """Convert channel indices to formatted string names."""
        hbo_list = []
        hbr_list = []
        nchan = len(ch_list)
        for i in range(0, nchan // 2):
            name = f"{ch_list[i][0]}_{ch_list[i][1]}_hbo"
            hbo_list.append(name)
        for i in range(nchan // 2, nchan):
            name = f"{ch_list[i][0]}_{ch_list[i][1]}_hbr"
            hbr_list.append(name)
        return hbo_list + hbr_list

    def trans_str_to_list(ch_names):
        """Convert formatted channel names to index lists."""
        return [[int(name.split('_')[0]), int(name.split('_')[1])] for name in ch_names]

    # Convert to optical density
    od_data = fnirs_trans_optical_density(data)

    # Apply wavelet denoising to each channel
    denoised_od = [wavelet_filter(od, wavelet=wavelet, level=level, alpha=alpha, percent=percent)
                   for od in od_data]

    # Apply bandpass filtering
    b, a = bessel_bandpass(lowcut=lowcut, highcut=highcut, fs=fs, filter_order=filter_order)
    filtered_data = signal.filtfilt(b, a, denoised_od)

    # Convert to hemoglobin concentrations
    nchan = len(channel_list)
    channel_indices = trans_str_to_list(channel_list)
    hbo, hbr = fnirs_beer_lambert_law(
        data=filtered_data,
        nchan=nchan,
        freqs=light_freqs,
        channel_list=channel_indices,
        src_pos=src_pos,
        det_pos=det_pos
    )

    # Prepare output structure
    result = {
        'data': np.vstack((hbo, hbr)).tolist(),
        'freqs': light_freqs,
        'srate': fs,
        'nchan': nchan,
        'ch_names': trans_list_to_str(channel_indices),
        'events': events,
        'type': 'fnirs_preprocess',
        'montage': None,
        'loc': [src_pos, det_pos]
    }

    # Save results if requested
    if is_save:
        save_file(data=result, save_path=save_path, save_filestyle=save_filestyle)

    return result


def fnirs_preprocessing_by_raw(raw, bad_channels, lowcut=0.01, highcut=0.7,
                               wavelet='db4', level=4, alpha=0.2,
                               enable_interpolate=False):
    """
    Preprocess fNIRS MNE Raw object including artifact removal and filtering.

    :param raw: MNE Raw object containing fNIRS data
    :type raw: mne.io.Raw
    :param bad_channels: Names of channels to exclude
    :type bad_channels: list
    :param lowcut: Low cutoff frequency (Hz, default: 0.01)
    :type lowcut: float
    :param highcut: High cutoff frequency (Hz, default: 0.7)
    :type highcut: float
    :param wavelet: Wavelet type for denoising (default: 'db4')
    :type wavelet: str
    :param level: Wavelet decomposition level (default: 4)
    :type level: int
    :param alpha: Wavelet threshold multiplier (default: 0.2)
    :type alpha: float
    :param enable_interpolate: Enable outlier interpolation flag (default: False)
    :type enable_interpolate: bool
    :return: Preprocessed haemodynamic data
    :rtype: mne.io.Raw
    """
    # Copy and remove bad channels
    raw_intensity = raw.copy()
    if bad_channels:
        raw_intensity.drop_channels(bad_channels)

    # Optical density conversion
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)

    # Beer-Lambert law conversion
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)

    # Bandpass filtering
    filtered_data = mne.filter.filter_data(
        raw_haemo.get_data(),
        raw_haemo.info['sfreq'],
        l_freq=lowcut,
        h_freq=highcut,
        method='fir'
    )
    raw_haemo._data = filtered_data

    # Wavelet denoising
    if wavelet:
        for ch in range(len(raw_haemo.info['ch_names'])):
            raw_haemo._data[ch] = wavelet_denoising(
                raw_haemo._data[ch],
                wavelet=wavelet,
                level=level,
                alpha=alpha
            )

    # Outlier detection and interpolation
    if enable_interpolate:
        for ch in range(len(raw_haemo.info['ch_names'])):
            raw_haemo._data[ch] = detect_and_interpolate_outliers(raw_haemo._data[ch])

    # Detrending
    raw_haemo._data = signal.detrend(raw_haemo.get_data(), axis=1)

    # Standardization
    raw_haemo._data = (raw_haemo.get_data() - np.mean(raw_haemo.get_data(), axis=1, keepdims=True)
                       ) / np.std(raw_haemo.get_data(), axis=1, keepdims=True)

    return raw_haemo


def wavelet_denoising(data, wavelet='db1', level=4, alpha=1):
    """
    Remove noise using wavelet thresholding techniques.

    :param data: Input 1D signal data
    :type data: numpy.ndarray
    :param wavelet: Wavelet type (default: 'db1')
    :type wavelet: str
    :param level: Decomposition level (default: 4)
    :type level: int
    :param alpha: Threshold multiplier (default: 1)
    :type alpha: float
    :return: Denoised signal data
    :rtype: numpy.ndarray
    """
    # Wavelet decomposition
    coeffs = pywt.wavedec(data, wavelet, level=level)

    # Coefficient thresholding
    for i in range(1, len(coeffs)):
        threshold = alpha * np.max(np.abs(coeffs[i]))
        coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')

    # Wavelet reconstruction
    denoised = pywt.waverec(coeffs, wavelet)

    # Length correction
    if len(denoised) > len(data):
        denoised = denoised[:len(data)]
    elif len(denoised) < len(data):
        padding = np.zeros(len(data) - len(denoised))
        denoised = np.concatenate((denoised, padding))

    return denoised


def fnirs_trans_optical_density(data):
    """
    Convert optical intensity data to optical density.

    :param data: Raw optical intensity data
    :type data: list or numpy.ndarray
    :return: Optical density converted data
    :rtype: list
    """
    od_data = []
    for sig in data:
        sig_mean = np.mean(sig)
        sig /= sig_mean
        sig = -np.log(sig)
        od_data.append(sig)
    return od_data


def fnirs_beer_lambert_law(data, nchan, freqs, channel_list, src_pos, det_pos, ppf=6.):
    """
    Convert optical density to oxy-/deoxy-haemoglobin concentrations.

    :param data: Optical density data
    :type data: list or numpy.ndarray
    :param nchan: Number of channels
    :type nchan: int
    :param freqs: Light wavelength frequencies [λ1, λ2] (nm)
    :type freqs: list
    :param channel_list: Source-detector pairing information
    :type channel_list: list
    :param src_pos: Source positions coordinates
    :type src_pos: list
    :param det_pos: Detector positions coordinates
    :type det_pos: list
    :param ppf: Partial pathlength factor (default: 6.0)
    :type ppf: float
    :return: Hemoglobin concentration arrays (HbO, HbR)
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """

    def calculate_distances(chan_list, src_pos, det_pos):
        """Compute source-detector distances for each channel."""
        distances = []
        for src_id, det_id in chan_list:
            dist = np.linalg.norm(np.array(src_pos[src_id - 1]) - np.array(det_pos[det_id - 1]))
            distances.append(dist)
        return distances

    def load_extinction_coefficients(freqs):
        """Load and interpolate extinction coefficients."""
        extinction_path = os.path.join(os.path.dirname(__file__), 'extinction_coef.mat')
        ext_data = scipy.io.loadmat(extinction_path)['extinct_coef']

        # Interpolate coefficients at specified wavelengths
        hbo_interp = interp1d(ext_data[:, 0], ext_data[:, 1], kind='linear')
        hb_interp = interp1d(ext_data[:, 0], ext_data[:, 2], kind='linear')

        return np.array([
            [hbo_interp(freqs[0]), hb_interp(freqs[0])],
            [hbo_interp(freqs[1]), hb_interp(freqs[1])]
        ]) * 0.2303  # Convert extinction to absorption

    abs_coef = load_extinction_coefficients(freqs)
    distances = calculate_distances(channel_list, src_pos, det_pos)

    hbo_concentrations = []
    hbr_concentrations = []

    for i in range(nchan // 2):
        # Calculate extinction matrix
        EL = abs_coef * distances[i] * ppf
        # Compute pseudo-inverse
        iEL = np.linalg.pinv(EL)
        # Convert to hemoglobin concentrations
        hb_levels = iEL @ data[[i, i + nchan // 2]] * 1e-3
        hbo_concentrations.append(hb_levels[0])
        hbr_concentrations.append(hb_levels[1])

    return np.array(hbo_concentrations), np.array(hbr_concentrations)


def ecg_preprocessing(ecg_signal, fs, lowcut, highcut, events=None, chan_list=None, downsample_fs=None,
                      filter_order=4, filter_method='Butterworth', notch_f=50, Q=30,
                      enable_baseline=True, median_window=0.2, enable_denoising=False,
                      wavelet='db4', level=1, is_save=False, save_path=None, save_filestyle='mat'):
    """
    Comprehensive ECG preprocessing pipeline with artifact removal options.

    :param ecg_signal: Raw ECG signal data
    :type ecg_signal: list or numpy.ndarray
    :param fs: Original sampling frequency (Hz)
    :type fs: float
    :param lowcut: Bandpass low cutoff frequency (Hz)
    :type lowcut: float
    :param highcut: Bandpass high cutoff frequency (Hz)
    :type highcut: float
    :param events: Event markers information
    :type events: list
    :param chan_list: Channel names list
    :type chan_list: list
    :param downsample_fs: Target sampling frequency (Hz, optional)
    :type downsample_fs: float
    :param filter_order: Bandpass filter order (default: 4)
    :type filter_order: int
    :param filter_method: Filter design method (default: 'Butterworth')
    :type filter_method: str
    :param notch_f: Notch frequency (default: 50 Hz)
    :type notch_f: float
    :param Q: Notch filter quality factor (default: 30)
    :type Q: float
    :param enable_baseline: Enable baseline correction flag (default: True)
    :type enable_baseline: bool
    :param median_window: Median filter window duration (seconds, default: 0.2)
    :type median_window: float
    :param enable_denoising: Enable wavelet denoising flag (default: False)
    :type enable_denoising: bool
    :param wavelet: Wavelet type for denoising (default: 'db4')
    :type wavelet: str
    :param level: Wavelet decomposition level (default: 1)
    :type level: int
    :param is_save: Save preprocessing results flag (default: False)
    :type is_save: bool
    :param save_path: Output file save path
    :type save_path: str
    :param save_filestyle: Output file format (default: 'mat')
    :type save_filestyle: str
    :return: Preprocessed ECG data structure
    :rtype: dict
    """
    # Bandpass filtering
    filtered = signal_filter(
        ecg_signal,
        fs=fs,
        lowcut=lowcut,
        highcut=highcut,
        filter_order=filter_order,
        method=filter_method
    )

    # Notch filtering
    filtered = notch_filter(filtered, fs=fs, f0=notch_f, Q=Q)

    # Optional downsampling
    if downsample_fs:
        filtered, fs = downsample(filtered, fs, downsample_fs)

    # Detrending
    detrended = signal_detrend(filtered)

    # Wavelet denoising
    if enable_denoising:
        denoised = wavelet_denoising(detrended, wavelet=wavelet, level=level)
    else:
        denoised = detrended

    # Median filtering
    if median_window:
        win_len = int(median_window * fs)
        win_len = win_len + 1 if win_len % 2 == 0 else win_len  # Ensure odd length
        median_filtered = median_filter(denoised, win_len)
    else:
        median_filtered = denoised

    # Baseline removal
    if enable_baseline:
        baseline_corrected = remove_baseline(median_filtered)
    else:
        baseline_corrected = median_filtered

    # Standardization
    standardized = standardize(baseline_corrected)

    # Determine channel count
    nchan = len(ecg_signal) if isinstance(ecg_signal, list) else ecg_signal.shape[0]
    chan_list = check_channel_list(chan_list, nchan)

    # Prepare output structure
    result = {
        'data': standardized,
        'srate': fs,
        'nchan': nchan,
        'ch_names': chan_list,
        'events': events,
        'type': 'ecg_preprocess',
        'montage': None
    }

    # Save results if requested
    if is_save:
        save_file(data=result, save_path=save_path, save_filestyle=save_filestyle)

    return result


def emg_preprocessing(emg_signal, fs, bf_lowcut, bf_highcut, lf_cutoff, events=None, chan_list=None,
                      bf_order=6, filter_method='Butterworth', lf_order=4,
                      notch_f=50, Q=30, is_save=False,
                      save_path=None, save_filestyle='mat'):
    """
    Comprehensive EMG preprocessing pipeline including rectification and filtering.

    :param emg_signal: Raw EMG signal data
    :type emg_signal: list or numpy.ndarray
    :param fs: Sampling frequency (Hz)
    :type fs: float
    :param bf_lowcut: Bandpass filter low cutoff frequency (Hz)
    :type bf_lowcut: float
    :param bf_highcut: Bandpass filter high cutoff frequency (Hz)
    :type bf_highcut: float
    :param lf_cutoff: Lowpass filter cutoff frequency (Hz)
    :type lf_cutoff: float
    :param events: Event markers information
    :type events: list
    :param chan_list: Channel names list
    :type chan_list: list
    :param bf_order: Bandpass filter order (default: 6)
    :type bf_order: int
    :param filter_method: Filter design method (default: 'Butterworth')
    :type filter_method: str
    :param lf_order: Lowpass filter order (default: 4)
    :type lf_order: int
    :param notch_f: Fundamental notch frequency (Hz, default: 50)
    :type notch_f: float
    :param Q: Notch filter quality factor (default: 30)
    :type Q: float
    :param is_save: Save preprocessing results flag (default: False)
    :type is_save: bool
    :param save_path: Output file save path
    :type save_path: str
    :param save_filestyle: Output file format (default: 'mat')
    :type save_filestyle: str
    :return: Preprocessed EMG data structure
    :rtype: dict
    """
    # Bandpass filtering
    bandpassed = signal_filter(
        emg_signal,
        fs=fs,
        lowcut=bf_lowcut,
        highcut=bf_highcut,
        filter_order=bf_order,
        method=filter_method
    )

    # Harmonic notch filtering
    notch_filtered = multi_notch_filter(bandpassed, fs=fs, f0=notch_f, Q=Q)

    # Full-wave rectification
    rectified = np.abs(notch_filtered)

    # Lowpass filtering
    lowpassed = butter_lowpass_filter(
        rectified,
        cutoff=lf_cutoff,
        fs=fs,
        order=lf_order
    )

    # Standardization
    standardized = standardize(lowpassed)

    # Determine channel count
    if isinstance(emg_signal, list):
        if is_multidimensional_list(emg_signal):
            standardized = standardized.reshape(1, -1)
        nchan = len(emg_signal)
    else:
        if emg_signal.ndim == 1:
            standardized = standardized.reshape(1, -1)
        nchan = emg_signal.shape[0]

    chan_list = check_channel_list(chan_list, nchan)

    # Prepare output structure
    result = {
        'data': standardized,
        'srate': fs,
        'nchan': nchan,
        'ch_names': chan_list,
        'events': events,
        'type': 'emg_preprocess',
        'montage': None
    }

    # Save results if requested
    if is_save:
        save_file(data=result, save_path=save_path, save_filestyle=save_filestyle)

    return result


def create_epoch(data, method, is_save, save_path, save_filestyle='mat',
                 events_list=None, events_range=None, fix_length=None,
                 custom_events=None, is_point=False):
    """
    Segment continuous data into epochs using various methods.

    :param data: Continuous physiological data structure
    :type data: dict
    :param method: Epoching method ('split_by_events', 'split_by_front_and_back_of_events', etc.)
    :type method: str
    :param is_save: Save segmented epochs flag
    :type is_save: bool
    :param save_path: Output directory for saving epochs
    :type save_path: str
    :param save_filestyle: Output file format (default: 'mat')
    :type save_filestyle: str
    :param events_list: Event timing information
    :type events_list: list
    :param events_range: Time window around events [pre, post] (samples)
    :type events_range: list
    :param fix_length: Fixed epoch duration (samples)
    :type fix_length: int
    :param custom_events: Custom event information
    :type custom_events: list
    :param is_point: Event marks are point events flag (default: False)
    :type is_point: bool
    :return: None (output saved to files)
    :rtype: None
    """
    epochs = []
    metadata = data.copy()
    metadata['data'] = None
    signal_data = np.array(data['data'])

    # Determine sampling rate
    srate = 1 if is_point else data['srate']

    if method == 'split_by_events' and events_list:
        # Segment data between consecutive events
        for i in range(len(events_list) - 1):
            start_sample = int(events_list[i] * srate)
            end_sample = int(events_list[i + 1] * srate)
            epochs.append(signal_data[:, start_sample:end_sample])

    elif method == 'split_by_front_and_back_of_events' and events_list and events_range:
        # Segment fixed windows around events
        for event_time in events_list:
            event_sample = int(event_time * srate)
            start = event_sample - events_range[0]
            end = event_sample + events_range[1]
            epochs.append(signal_data[:, start:end])

    elif method == 'split_by_fixed_length' and fix_length:
        # Segment data into fixed-length epochs
        n_epochs = signal_data.shape[1] // fix_length
        for i in range(n_epochs):
            start = i * fix_length
            epochs.append(signal_data[:, start:start + fix_length])

    elif method == 'custom_split':
        # Custom epoching logic (user-defined)
        pass

    # Save segmented epochs
    if is_save:
        os.makedirs(save_path, exist_ok=True)
        for i, epoch_data in enumerate(epochs):
            epoch_meta = metadata.copy()
            epoch_meta['data'] = epoch_data
            epoch_meta['events'] = None
            save_file(
                data=epoch_meta,
                save_path=os.path.join(save_path, f"{i + 1}.{save_filestyle}"),
                save_filestyle=save_filestyle
            )
