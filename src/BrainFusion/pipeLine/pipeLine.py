# -*- coding: utf-8 -*-
# @Time    : 2024/2/29 15:58
# @Author  : Li WenHao
# @Site    : South China University of Technology
# @File    : pipeLine.py
# @Software: PyCharm 
# @Comment :
import os.path
import EntropyHub
import mne
import networkx as nx
import numpy as np
import pywt
from matplotlib import pyplot as plt
from scipy.signal import find_peaks, spectrogram, coherence
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score
from BrainFusion.io.File_IO import save_file, save_feature
from fooof import FOOOF
from fooof.utils import trim_spectrum
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm.first_level.first_level import run_glm
import neurokit2 as nk
from statsmodels.tsa.stattools import grangercausalitytests
from BrainFusion.utils.channels import convert_ndarray_to_list


def eeg_power_spectral_density(data, is_save, save_path=None, save_filestyle='csv'):
    """
    Calculate Power Spectral Density (PSD) and extract band powers

    :param data: EEG data dictionary with keys
    :type data: dict
    :param is_save: Save results flag
    :type is_save: bool
    :param save_path: Output file path
    :type save_path: str, optional
    :param save_filestyle: Output file format
    :type save_filestyle: str
    :return: DataFrame with PSD results
    :rtype: pd.DataFrame
    """
    if data:
        # Define frequency band boundaries
        delta_band = (0.5, 4)
        theta_band = (4, 8)
        alpha_band = (8, 13)
        beta_band = (13, 30)
        gamma_band = (31, 40)

        average_powers = []
        channel_names = data['ch_names']
        NFFT = 2 ** int(np.ceil(np.log2(data['srate'])))

        # Process each channel
        for i, channel_data in enumerate(data['data']):
            # Calculate power spectrum
            power_spectrum, freq = plt.psd(channel_data, NFFT=NFFT, Fs=data['srate'])

            # Calculate band powers
            delta_power = np.trapz(power_spectrum[(freq >= delta_band[0]) & (freq <= delta_band[1])])
            theta_power = np.trapz(power_spectrum[(freq >= theta_band[0]) & (freq <= theta_band[1])])
            alpha_power = np.trapz(power_spectrum[(freq >= alpha_band[0]) & (freq <= alpha_band[1])])
            beta_power = np.trapz(power_spectrum[(freq >= beta_band[0]) & (freq <= beta_band[1])])
            gamma_power = np.trapz(power_spectrum[(freq >= gamma_band[0]) & (freq <= gamma_band[1])])

            # Store results
            average_powers.append([channel_names[i], delta_power, theta_power, alpha_power, beta_power, gamma_power])

        columns = ['Channel'] + ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        psd_df = pd.DataFrame(average_powers, columns=columns)
        psd_df['Type'] = 'eeg_psd'

        if is_save:
            save_feature(data=psd_df, save_path=save_path, save_filestyle=save_filestyle)

        return psd_df
    else:
        return None


def power_spectral_density(data, selected_band, is_save, save_path=None, save_filestyle='csv'):
    """
    Calculate custom power spectral density for selected bands

    :param data: Signal data dictionary with keys
    :type data: dict
    :param selected_band: List of frequency bands to analyze
    :type selected_band: list
    :param is_save: Save results flag
    :type is_save: bool
    :param save_path: Output file path
    :type save_path: str, optional
    :param save_filestyle: Output file format
    :type save_filestyle: str
    :return: Processed PSD data
    :rtype: dict
    """
    if data:
        average_powers = []
        # Calculate FFT size
        NFFT = 2 ** int(np.ceil(np.log2(data['srate'])))

        # Process each channel
        for channel_data in data['data']:
            # Calculate power spectrum
            power_spectrum, freq = plt.psd(channel_data, NFFT=NFFT, Fs=data['srate'])

            average_band_powers = []
            # Calculate power for each band
            for band in selected_band:
                power = np.trapz(power_spectrum[(freq >= band[0]) & (freq <= band[1])])
                average_band_powers.append(power)

            # Store results
            average_powers.append(average_band_powers)

        # Format results
        psd = {}
        psd['data'] = average_powers
        psd['srate'] = data['srate']
        psd['nchan'] = data['nchan']
        psd['ch_names'] = data['ch_names']
        psd['events'] = None
        psd['type'] = 'psd'
        psd['montage'] = data['montage']

        if is_save:
            save_file(data=psd, save_path=save_path, save_filestyle=save_filestyle)

        return psd
    else:
        return None


def sample_entropy(data, is_save, save_path=None, save_filestyle='mat'):
    """
    Calculate Sample Entropy of the signal

    :param data: Signal data dictionary with keys
    :type data: dict
    :param is_save: Save results flag
    :type is_save: bool
    :param save_path: Output file path
    :type save_path: str, optional
    :param save_filestyle: Output file format
    :type save_filestyle: str
    :return: Sample Entropy results
    :rtype: pd.DataFrame
    """
    if data:
        entropy_list = []
        # Calculate sample entropy for each channel
        for i, array in enumerate(data['data']):
            sample_entropy = EntropyHub.SampEn(array, m=2, r=0.2 * np.std(array))[0][-1]
            entropy_list.append([data['ch_names'][i], sample_entropy])

        # Format results
        columns = ['Channel', 'Sample Entropy']
        sample_entropy_df = pd.DataFrame(entropy_list, columns=columns)
        sample_entropy_df['Type'] = 'sample_entropy'

        if is_save:
            save_feature(data=sample_entropy_df, save_path=save_path, save_filestyle=save_filestyle)

        return sample_entropy_df
    else:
        return None


def coarse_graining(signal, scale_factor):
    """
    Perform coarse-graining on signal

    :param signal: Input signal vector
    :type signal: np.ndarray
    :param scale_factor: Coarse-graining factor
    :type scale_factor: int
    :return: Coarse-grained signal
    :rtype: np.ndarray
    """
    coarse_grained_signal = [np.mean(signal[i:i + scale_factor]) for i in range(0, len(signal), scale_factor)]
    return np.array(coarse_grained_signal)


def multiscale_sample_entropy(signal, max_scale_factor):
    """
    Calculate Multiscale Sample Entropy (MSE)

    :param signal: Input signal vector
    :type signal: np.ndarray
    :param max_scale_factor: Maximum scale factor
    :type max_scale_factor: int
    :return: Entropy values at each scale
    :rtype: list
    """
    entropies = []
    for scale_factor in range(1, max_scale_factor + 1):
        # Apply coarse-graining
        coarse_signal = coarse_graining(signal, scale_factor)
        # Calculate sample entropy
        sample_entropy = EntropyHub.SampEn(coarse_signal, m=2, r=0.2 * np.std(signal))[0][-1]
        entropies.append(sample_entropy)
    return entropies


def eeg_multiscale_entropy(data, is_save, scale_factor=1, save_path=None, save_filestyle='csv'):
    """
    Calculate Multiscale Entropy for EEG signals

    :param data: EEG data dictionary with keys
    :type data: dict
    :param is_save: Save results flag
    :type is_save: bool
    :param scale_factor: Scale factor for analysis
    :type scale_factor: int
    :param save_path: Output file path
    :type save_path: str, optional
    :param save_filestyle: Output file format
    :type save_filestyle: str
    :return: Multiscale entropy results
    :rtype: pd.DataFrame
    """
    multiscale_entropy_list = []
    for i, array in enumerate(data['data']):
        # Calculate multiscale entropy
        multiscale_entropy = multiscale_sample_entropy(array, scale_factor)
        channel_name = data['ch_names'][i]
        multiscale_entropy_list.append([channel_name] + multiscale_entropy)

    # Format results
    columns = ['Channel'] + [f'Scale_{i}' for i in range(1, len(multiscale_entropy_list[0]))]
    multiscale_entropy_df = pd.DataFrame(multiscale_entropy_list, columns=columns)
    multiscale_entropy_df['Type'] = 'multiscale_entropy'

    if is_save:
        save_feature(data=multiscale_entropy_df, save_path=save_path, save_filestyle=save_filestyle)

    return multiscale_entropy_df


def eeg_microstate(data, n_clusters=4, peak_threshold=None, is_show=True, is_save=False, save_path=None,
                   save_filestyle='csv'):
    """
    Perform EEG microstate analysis using GFP peaks

    :param data: EEG data dictionary with keys
    :type data: dict
    :param n_clusters: Number of microstate classes
    :type n_clusters: int
    :param peak_threshold: GFP peak detection threshold
    :type peak_threshold: float, optional
    :param is_show: Show visualization flag
    :type is_show: bool
    :param is_save: Save results flag
    :type is_save: bool
    :param save_path: Output file path
    :type save_path: str, optional
    :param save_filestyle: Output file format
    :type save_filestyle: str
    :return: Microstate cluster centroids
    :rtype: pd.DataFrame
    """

    def min_max_scaling_to_range(array, new_min=-1, new_max=1):
        """
        Min-max scaling to specified range

        :param array: Input array
        :type array: np.ndarray
        :param new_min: Minimum value of new range
        :type new_min: float
        :param new_max: Maximum value of new range
        :type new_max: float
        :return: Normalized array
        :rtype: np.ndarray
        """
        min_val = np.min(array)
        max_val = np.max(array)
        normalized_array = new_min + (new_max - new_min) * (array - min_val) / (max_val - min_val)
        return normalized_array

    def min_max_scaling_by_arrays(arrays, new_min=-1, new_max=1):
        """
        Apply min-max scaling to multiple arrays

        :param arrays: List of input arrays
        :type arrays: list
        :param new_min: Minimum value of new range
        :type new_min: float
        :param new_max: Maximum value of new range
        :type new_max: float
        :return: Normalized arrays
        :rtype: np.ndarray
        """
        normalized_arrays = []
        for array in arrays:
            normalized_array = min_max_scaling_to_range(array, new_min, new_max)
            normalized_arrays.append(normalized_array)
        return np.array(normalized_arrays)

    # Calculate Global Field Power (GFP)
    signal_data = np.array(data['data'])
    gfp = np.std(signal_data, axis=0)

    # Detect GFP peaks
    gfp_peaks_indices, _ = find_peaks(gfp, height=peak_threshold)
    gfp_peaks_times = gfp_peaks_indices / data['srate']  # Convert indices to seconds

    # Cluster EEG topography at peak points
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(signal_data[:, gfp_peaks_indices].T)
    cluster_centers = kmeans.cluster_centers_

    # Classify entire EEG time series
    labels = kmeans.predict(signal_data.T)
    print("Cluster Labels:", labels)

    # Visualization if requested
    if is_show:
        # Plot GFP with peaks
        length = np.array(signal_data).shape[1]
        plt.plot(np.arange(length) / data['srate'], gfp)
        plt.scatter(gfp_peaks_times, gfp[gfp_peaks_indices], c='red', marker='^', label='GFP Peaks')
        plt.xlabel('Time (s)')
        plt.ylabel('GFP')
        plt.legend()
        plt.show()

        # Create microstate topography plots
        fig, axes = plt.subplots(1, n_clusters, figsize=(n_clusters * 2, 6), sharex=True, sharey=False)
        fig.subplots_adjust(hspace=0, wspace=0.05, bottom=0.08, left=0.05, top=0.98, right=0.98)
        fig.suptitle("EEG Microstate")

        # Prepare MNE objects for topography plotting
        montage = mne.channels.make_standard_montage('standard_1020')
        info = mne.create_info(ch_names=data['ch_names'], sfreq=data['srate'], ch_types='eeg')
        norm_data = min_max_scaling_by_arrays(cluster_centers)
        data_range = (-1, 1)

        # Plot each microstate topography
        for i, state in enumerate(norm_data):
            evoked = mne.EvokedArray(data=np.array(norm_data.T), info=info)
            evoked.set_montage(montage)
            axes[i].clear()
            mne.viz.plot_topomap(
                state, evoked.info,
                axes=axes[i], show=False,
                sensors=True, vlim=data_range
            )
            axes[i].figure.canvas.draw()
        plt.show()

    # Format results as DataFrame
    microstate_df = pd.DataFrame(cluster_centers.T, columns=[f'Cluster_{i + 1}' for i in range(n_clusters)])
    microstate_df.insert(0, 'Channel', data['ch_names'])
    microstate_df['Number of Clusters'] = n_clusters
    microstate_df['Type'] = 'eeg_microstate'
    microstate_df.reset_index(drop=True, inplace=True)

    if is_save:
        save_feature(data=microstate_df, save_path=save_path, save_filestyle=save_filestyle)

    return microstate_df


def root_mean_square(data, is_save, is_sliding=False, window_size=None, overlap_rate=0, save_path=None,
                     save_filestyle='mat'):
    """
    Calculate Root Mean Square (RMS) value

    :param data: Signal data dictionary with keys
    :type data: dict
    :param is_save: Save results flag
    :type is_save: bool
    :param is_sliding: Use sliding window flag
    :type is_sliding: bool
    :param window_size: Sliding window size
    :type window_size: int, optional
    :param overlap_rate: Window overlap rate
    :type overlap_rate: float, optional
    :param save_path: Output file path
    :type save_path: str, optional
    :param save_filestyle: Output file format
    :type save_filestyle: str
    :return: RMS values
    :rtype: pd.DataFrame
    """
    signal_data = np.array(data['data'])
    # Calculate RMS across time dimension
    rms_values = np.sqrt(np.mean(signal_data ** 2, axis=1))

    # Format results
    rms_df = pd.DataFrame({'Channel': data['ch_names']})
    rms_df['rms'] = rms_values.T.tolist()
    rms_df['Type'] = 'rms'

    if is_save:
        save_feature(data=rms_df, save_path=save_path, save_filestyle=save_filestyle)

    return rms_df


def variance(data, is_save, is_sliding=False, window_size=None, overlap_rate=0, save_path=None,
             save_filestyle='mat'):
    """
    Calculate signal variance

    :param data: Signal data dictionary with keys
    :type data: dict
    :param is_save: Save results flag
    :type is_save: bool
    :param is_sliding: Use sliding window flag
    :type is_sliding: bool
    :param window_size: Sliding window size
    :type window_size: int, optional
    :param overlap_rate: Window overlap rate
    :type overlap_rate: float, optional
    :param save_path: Output file path
    :type save_path: str, optional
    :param save_filestyle: Output file format
    :type save_filestyle: str
    :return: Variance values
    :rtype: pd.DataFrame
    """
    signal_data = np.array(data['data'])
    # Calculate variance across time dimension
    var_values = np.var(signal_data, axis=1)

    # Format results
    var_df = pd.DataFrame({'Channel': data['ch_names']})
    var_df['rms'] = var_values.T.tolist()
    var_df['Type'] = 'var'

    if is_save:
        save_feature(data=var_df, save_path=save_path, save_filestyle=save_filestyle)

    return var_df


def mean_absolute_value(data, is_save, is_sliding=False, window_size=None, overlap_rate=0, save_path=None,
                        save_filestyle='csv'):
    """
    Calculate Mean Absolute Value (MAV)

    :param data: Signal data dictionary with keys
    :type data: dict
    :param is_save: Save results flag
    :type is_save: bool
    :param is_sliding: Use sliding window flag
    :type is_sliding: bool
    :param window_size: Sliding window size
    :type window_size: int, optional
    :param overlap_rate: Window overlap rate
    :type overlap_rate: float, optional
    :param save_path: Output file path
    :type save_path: str, optional
    :param save_filestyle: Output file format
    :type save_filestyle: str
    :return: MAV values
    :rtype: pd.DataFrame
    """
    signal_data = np.array(data['data'])
    # Calculate MAV across time dimension
    mav_values = np.mean(np.abs(signal_data), axis=1)

    # Format results
    mav_df = pd.DataFrame({'Channel': data['ch_names']})
    mav_df['rms'] = mav_values.T.tolist()
    mav_df['Type'] = 'mav'

    if is_save:
        save_feature(data=mav_df, save_path=save_path, save_filestyle=save_filestyle)

    return mav_df


def zero_crossing(data, is_save, is_sliding=False, window_size=None, overlap_rate=0, save_path=None,
                  save_filestyle='csv'):
    """
    Calculate Zero Crossing Rate (ZCR)

    :param data: Signal data dictionary with keys
    :type data: dict
    :param is_save: Save results flag
    :type is_save: bool
    :param is_sliding: Use sliding window flag
    :type is_sliding: bool
    :param window_size: Sliding window size
    :type window_size: int, optional
    :param overlap_rate: Window overlap rate
    :type overlap_rate: float, optional
    :param save_path: Output file path
    :type save_path: str, optional
    :param save_filestyle: Output file format
    :type save_filestyle: str
    :return: ZCR values
    :rtype: pd.DataFrame
    """

    def count_zero_crossings(signal):
        """
        Count zero crossings in a signal

        :param signal: Input signal vector
        :type signal: np.ndarray
        :return: Number of zero crossings
        :rtype: int
        """
        zero_crossings = np.where(np.diff(np.sign(signal)))[0]
        return len(zero_crossings)

    signal_data = np.array(data['data'])
    # Calculate ZCR for each channel
    zc_values = np.array([count_zero_crossings(channel_data) for channel_data in signal_data])

    # Format results
    zc_df = pd.DataFrame({'Channel': data['ch_names']})
    zc_df['rms'] = zc_values.T.tolist()
    zc_df['Type'] = 'zc'

    if is_save:
        save_feature(data=zc_df, save_path=save_path, save_filestyle=save_filestyle)

    return zc_df


def wavelet_transform(data, level, basis_function='db1', is_save=False, save_path=None,
                      save_filestyle='mat'):
    """
    Perform Discrete Wavelet Transform (DWT) on signal data. Applies wavelet decomposition to each channel and returns the coefficients

    :param data: Signal data dictionary with keys
    :type data: dict
    :param level: Decomposition level
    :type level: int
    :param basis_function: Wavelet basis function (default 'db1')
    :type basis_function: str
    :param is_save: Save results flag
    :type is_save: bool
    :param save_path: Output file path
    :type save_path: str, optional
    :param save_filestyle: Output file format
    :type save_filestyle: str
    :return: Wavelet coefficients
    :rtype: pd.DataFrame
    """
    coeffs_list = []
    # Perform wavelet decomposition for each channel
    for signal in data['data']:
        coeffs = pywt.wavedec(signal, basis_function, level=level)
        if isinstance(coeffs, np.ndarray):
            coeffs_list.append(coeffs.tolist())  # Convert NumPy array to list
        else:
            coeffs_list.append(coeffs)

    # Format results as DataFrame
    wt_df = pd.DataFrame({'Channel': data['ch_names']})
    # Convert coefficients list to a comma-separated string
    coeffs_list = convert_ndarray_to_list(coeffs_list)
    coeffs_str = [','.join(coeffs) for coeffs in coeffs_list]
    wt_df['Coefficients'] = coeffs_str
    wt_df['Type'] = 'dwt'
    wt_df['Level'] = level
    wt_df['Basis Function'] = basis_function

    if is_save:
        save_feature(data=wt_df, save_path=save_path, save_filestyle=save_filestyle)

    return wt_df


def wavelet_packet_energy(data, level, basis_function='db1', mode='symmetric', is_save=False, save_path=None,
                          save_filestyle='mat'):
    """
    Calculate wavelet packet energy for signal data

    :param data: Signal data dictionary with keys
    :type data: dict
    :param level: Decomposition level
    :type level: int
    :param basis_function: Wavelet basis function (default 'db1')
    :type basis_function: str
    :param mode: Extension mode (default 'symmetric')
    :type mode: str
    :param is_save: Save results flag
    :type is_save: bool
    :param save_path: Output file path
    :type save_path: str, optional
    :param save_filestyle: Output file format
    :type save_filestyle: str
    :return: Wavelet packet energy results
    :rtype: pd.DataFrame
    """
    wp_energy = []
    # Process each channel
    for signal in data['data']:
        wp = pywt.WaveletPacket(data=signal, wavelet=basis_function, mode='symmetric', maxlevel=level)
        # Compute energy for each decomposition node
        energy_per_band = [np.linalg.norm(node.data, ord=None) ** 2 for node in wp.get_level(wp.maxlevel, "freq")]
        wp_energy.append(energy_per_band)

    # Format energy results as strings
    energy_str = [','.join(map(str, band)) for band in wp_energy]

    wpe_df = pd.DataFrame({'Channel': data['ch_names']})
    wpe_df['Energy'] = energy_str
    wpe_df['Type'] = 'wpe'
    wpe_df['Level'] = level
    wpe_df['Basis Function'] = basis_function

    if is_save:
        save_feature(data=wpe_df, save_path=save_path, save_filestyle=save_filestyle)

    return wpe_df


def continuous_wavelet_transform(data, widths=np.arange(1, 50), basis_function='cmor', is_save=False, save_path=None,
                                 save_filestyle='mat'):
    """
    Compute Continuous Wavelet Transform (CWT) for signal data

    :param data: Signal data dictionary with keys
    :type data: dict
    :param widths: Array of scales for wavelet transform
    :type widths: np.ndarray
    :param basis_function: Wavelet basis function (default 'cmor')
    :type basis_function: str
    :param is_save: Save results flag
    :type is_save: bool
    :param save_path: Output file path
    :type save_path: str, optional
    :param save_filestyle: Output file format
    :type save_filestyle: str
    :return: CWT results
    :rtype: pd.DataFrame
    """
    cwt_list = []
    freqs_list = []
    time_list = []
    ch_names = data['ch_names']

    # Compute CWT for each channel
    for signal in np.array(data['data']):
        t = np.arange(0, signal.shape[0])
        # Perform continuous wavelet transform
        cwt_matrix, frequencies = pywt.cwt(signal, widths, basis_function, sampling_period=1.0 / data['srate'])
        # Format results as strings
        cwt_matrix_str = [','.join(map(str, row)) for row in cwt_matrix]
        cwt_list.append(cwt_matrix_str)
        freqs_list.append(','.join(map(str, frequencies)))
        time_list.append(','.join(map(str, t)))

    # Format results as DataFrame
    cwt_df = pd.DataFrame({
        'Channel': ch_names,
        'Frequencies': freqs_list,
        'Time': time_list,
        'CWT Data': cwt_list,
        'Type': 'cwt',
        'Basis Function': basis_function
    })

    if is_save:
        save_feature(data=cwt_df, save_path=save_path, save_filestyle=save_filestyle)

    return cwt_df


def Hillport_Huang_transform(data, threshold, is_save, save_path=None, save_filestyle='csv'):
    """
    Placeholder for Hillport-Huang Transform

    :param data: Input data
    :type data: dict
    :param threshold: Threshold parameter
    :type threshold: float
    :param is_save: Save results flag
    :type is_save: bool
    :param save_path: Output file path
    :type save_path: str, optional
    :param save_filestyle: Output file format
    :type save_filestyle: str
    """
    pass


def short_time_Fourier_transform(data, nperseg, noverlap, window_method='hamming', is_save=False, save_path=None,
                                 save_filestyle='mat'):
    """
    Compute Short-Time Fourier Transform (STFT) for signal data

    :param data: Signal data dictionary with keys
    :type data: dict
    :param nperseg: Length of each segment
    :type nperseg: int
    :param noverlap: Overlap between segments
    :type noverlap: int
    :param window_method: Window function (default 'hamming')
    :type window_method: str
    :param is_save: Save results flag
    :type is_save: bool
    :param save_path: Output file path
    :type save_path: str, optional
    :param save_filestyle: Output file format
    :type save_filestyle: str
    :return: STFT results
    :rtype: pd.DataFrame
    """
    stft_list = []
    frequencies_list = []
    times_list = []
    ch_names = data['ch_names']

    # Compute STFT for each channel
    for signal in data['data']:
        signal = np.array(signal)
        # Compute spectrogram (STFT)
        frequencies, times, Sxx = spectrogram(signal, fs=data['srate'], nperseg=nperseg, noverlap=noverlap,
                                              window=window_method)
        # Format results as strings
        Sxx_str = [','.join(map(str, row)) for row in Sxx]
        stft_list.append(Sxx_str)
        frequencies_list.append(','.join(map(str, frequencies)))
        times_list.append(','.join(map(str, times)))

    # Format results as DataFrame
    stft_df = pd.DataFrame({
        'Channel': ch_names,
        'Frequencies': frequencies_list,
        'Times': times_list,
        'STFT Data': stft_list,
        'Type': 'stft',
        'Window Method': window_method
    })

    if is_save:
        save_feature(data=stft_df, save_path=save_path, save_filestyle=save_filestyle)

    return stft_df


def network_adjacency_matrix(data, edge_retention_rate=1, is_relative_thresholds=True, threshold=0.5, method='cov'):
    """
    Compute network adjacency matrix

    :param data: Signal data dictionary with key 'data'
    :type data: dict
    :param edge_retention_rate: Proportion of edges to retain
    :type edge_retention_rate: float
    :param is_relative_thresholds: Use relative thresholding
    :type is_relative_thresholds: bool
    :param threshold: Connection threshold
    :type threshold: float
    :param method: Connectivity method ('cov' for covariance)
    :type method: str
    :return: Adjacency matrix
    :rtype: np.ndarray
    """
    adjacency_matrix = None
    signal_data = np.array(data['data'])

    # Compute adjacency matrix using covariance
    if method == 'cov':
        adjacency_matrix = np.corrcoef(signal_data)

    # Binarize adjacency matrix
    if adjacency_matrix is not None:
        if not is_relative_thresholds:
            flattened_adjacency = np.abs(adjacency_matrix.flatten())
            num_edges = len(flattened_adjacency)
            num_edges_to_keep = int(edge_retention_rate * num_edges)
            # Determine threshold for edge retention
            threshold_value = np.sort(flattened_adjacency)[-num_edges_to_keep]
            # Binarize based on threshold
            adjacency_matrix = np.where(np.abs(adjacency_matrix) >= threshold_value, 1, 0)
        else:
            adjacency_matrix = np.where(np.abs(adjacency_matrix) > threshold, 1, 0)

    return adjacency_matrix


def local_network_features(data, edge_retention_rate=1, is_relative_thresholds=True, threshold=0.5, method='cov',
                           is_save=False, save_path=None, save_filestyle='mat'):
    """
    Calculate local network properties

    :param data: Signal data dictionary with keys
    :type data: dict
    :param edge_retention_rate: Proportion of edges to retain
    :type edge_retention_rate: float
    :param is_relative_thresholds: Use relative thresholding
    :type is_relative_thresholds: bool
    :param threshold: Connection threshold
    :type threshold: float
    :param method: Connectivity method ('cov' for covariance)
    :type method: str
    :param is_save: Save results flag
    :type is_save: bool
    :param save_path: Output file path
    :type save_path: str, optional
    :param save_filestyle: Output file format
    :type save_filestyle: str
    :return: Local network properties
    :rtype: pd.DataFrame
    """
    # Compute adjacency matrix
    adjacency_matrix = network_adjacency_matrix(data, edge_retention_rate, is_relative_thresholds, threshold, method)
    # Create graph from adjacency matrix
    G = nx.from_numpy_array(adjacency_matrix)

    # Calculate graph properties
    degrees = list(dict(G.degree()).values())
    clustering_coefficients = list(nx.clustering(G).values())
    centrality = list(nx.degree_centrality(G).values())
    communities = list(nx.community.greedy_modularity_communities(G))
    communities = [list(set_x) for set_x in communities]

    # Format results as DataFrame
    df = pd.DataFrame({
        'Channel': data['ch_names'],
        'Degree': degrees,
        'Clustering Coefficient': clustering_coefficients,
        'Centrality': centrality,
        'Type': ['local_network'] * data['nchan']
    })

    if is_save:
        save_feature(data=df, save_path=save_path, save_filestyle=save_filestyle)

    return df


def global_network_features(data, edge_retention_rate=1, is_relative_thresholds=True, threshold=0.5, method='cov',
                            is_save=False, save_path=None, save_filestyle='mat'):
    """
    Calculate global network properties

    :param data: Signal data dictionary with keys
    :type data: dict
    :param edge_retention_rate: Proportion of edges to retain
    :type edge_retention_rate: float
    :param is_relative_thresholds: Use relative thresholding
    :type is_relative_thresholds: bool
    :param threshold: Connection threshold
    :type threshold: float
    :param method: Connectivity method ('cov' for covariance)
    :type method: str
    :param is_save: Save results flag
    :type is_save: bool
    :param save_path: Output file path
    :type save_path: str, optional
    :param save_filestyle: Output file format
    :type save_filestyle: str
    :return: Global network properties
    :rtype: pd.DataFrame
    """
    # Compute adjacency matrix
    adjacency_matrix = network_adjacency_matrix(data, edge_retention_rate, is_relative_thresholds, threshold, method)
    # Create graph from adjacency matrix
    G = nx.from_numpy_array(adjacency_matrix)

    # Calculate global graph properties
    average_shortest_path_length = nx.average_shortest_path_length(G)
    network_diameter = nx.diameter(G)
    average_clustering_coefficient = nx.average_clustering(G)
    network_density = nx.density(G)
    assortativity = nx.degree_assortativity_coefficient(G)
    connected_components = nx.number_connected_components(G)

    # Format results as DataFrame
    df = pd.DataFrame({
        'Type': ['global_network'],
        'Average Shortest Path Length': [average_shortest_path_length],
        'Diameter': [network_diameter],
        'Average Clustering Coefficient': [average_clustering_coefficient],
        'Density': [network_density],
        'Assortativity': [assortativity],
        'Connected Components': [connected_components]
    })

    if is_save:
        save_feature(data=df, save_path=save_path, save_filestyle=save_filestyle)

    return df


def hjorth_parameters(data, is_save=False, is_sliding=False, window_size=None, overlap_rate=0, save_path=None,
                      save_filestyle='mat'):
    """
    Calculate Hjorth parameters (Activity, Mobility, Complexity)

    :param data: Signal data dictionary with keys
    :type data: dict
    :param is_save: Save results flag
    :type is_save: bool
    :param is_sliding: Use sliding window flag
    :type is_sliding: bool
    :param window_size: Sliding window size
    :type window_size: int, optional
    :param overlap_rate: Window overlap rate
    :type overlap_rate: float, optional
    :param save_path: Output file path
    :type save_path: str, optional
    :param save_filestyle: Output file format
    :type save_filestyle: str
    :return: Hjorth parameters for each channel
    :rtype: pd.DataFrame
    """
    df = None

    def hjorth_activity(sig):
        """Calculate activity parameter"""
        activity = np.var(sig)
        return activity

    def hjorth_mobility(sig):
        """Calculate mobility parameter"""
        diff_sig = np.diff(sig)
        mobility = np.sqrt(np.var(diff_sig) / np.var(sig))
        return mobility

    def hjorth_complexity(sig):
        """Calculate complexity parameter"""
        diff_sig = np.diff(sig)
        diff_diff_sig = np.diff(diff_sig)
        complexity = np.sqrt(np.var(diff_diff_sig) / np.var(diff_sig)) / hjorth_mobility(sig)
        return complexity

    def hjorth(signal):
        """Calculate Hjorth parameters for multiple signals"""
        num_signals = signal.shape[0]
        activity_list = []
        mobility_list = []
        complexity_list = []

        for i in range(num_signals):
            sig = signal[i, :]
            activity = hjorth_activity(sig)
            mobility = hjorth_mobility(sig)
            complexity = hjorth_complexity(sig)

            activity_list.append(activity)
            mobility_list.append(mobility)
            complexity_list.append(complexity)

        return activity_list, mobility_list, complexity_list

    signal_data = np.array(data['data'])
    hjorth_values = hjorth(signal_data)
    activity_result = hjorth_values[0]
    mobility_result = hjorth_values[1]
    complexity_result = hjorth_values[2]

    # Format results as DataFrame
    df = pd.DataFrame({
        'Channel': data['ch_names'],
        'Activity': activity_result,
        'Mobility': mobility_result,
        'Complexity': complexity_result,
        'Type': ['hjorth'] * len(data['ch_names']),
    })

    if is_save:
        save_feature(data=df, save_path=save_path, save_filestyle=save_filestyle)

    return df


def aperiodic_parameters(data, is_save, save_path=None, save_filestyle='mat'):
    """
    Extract aperiodic components using FOOOF model

    :param data: EEG data dictionary with keys
    :type data: dict
    :param is_save: Save results flag
    :type is_save: bool
    :param save_path: Output file path
    :type save_path: str, optional
    :param save_filestyle: Output file format
    :type save_filestyle: str
    :return: FOOOF analysis results
    :rtype: dict
    """

    def iaf_fooof(signal, fs=1000):
        """Perform individual alpha frequency analysis using FOOOF"""
        psd, freqs = mne.time_frequency.psd_array_welch(signal, sfreq=fs, n_overlap=50, n_per_seg=100,
                                                        fmin=2, fmax=45,
                                                        n_fft=1000)
        full_spectrum_power_log = abs(np.mean(psd))
        fm = FOOOF(peak_width_limits=[2, 8], max_n_peaks=6, peak_threshold=1.5, aperiodic_mode='fixed')
        fm.fit(freqs, psd, freq_range=[2, 45])
        r2 = fm.r_squared_  # Model fit quality
        periodic_data = fm._peak_fit  # Periodic component
        aperiodic_data = fm._ap_fit  # Aperiodic component
        fooofed_data = fm.fooofed_spectrum_  # Full model spectrum
        center_frequency = fm.peak_params_[:, 0]  # Center frequencies
        power = fm.peak_params_[:, 1]  # Power values
        peak_width = fm.peak_params_[:, 2]  # Bandwidths

        # Extract alpha band parameters
        alpha_freqs, alpha_psd = trim_spectrum(freqs, psd, [7, 13])
        alpha_cf = None

        if np.logical_and(center_frequency > 7, center_frequency < 14).any():
            alpha_index = np.where(np.logical_and(center_frequency > 7, center_frequency < 14))
            if len(alpha_index[0]) > 1:
                indexs = np.where(power[alpha_index[0]] == np.max(power[alpha_index[0]]))
                index_alpha = indexs[0][0]
                index = alpha_index[0][index_alpha]
                alpha_cf = center_frequency[index]
            else:
                alpha_cf = center_frequency[alpha_index]

        if alpha_cf:
            difference_alpha_freqs = np.abs(alpha_freqs - alpha_cf)
            icf_index = np.argmin(difference_alpha_freqs)
            alpha_icf = alpha_freqs[icf_index]
            al_iamp = alpha_psd[icf_index]

            # Calculate individualized alpha power metrics
            lower_freq = alpha_icf - 4
            lower_freq_index = np.argwhere(freqs == lower_freq)[0, 0]
            upper_freq = alpha_icf + 2
            upper_freq_index = np.argwhere(freqs == upper_freq)[0, 0]
            indivial_alpha_power = abs(np.nanmean(psd[lower_freq_index:upper_freq_index + 1]))
            indivial_alpha_relpower = indivial_alpha_power / full_spectrum_power_log
            adjust_psd = fm.fooofed_spectrum_ - fm._ap_fit
            indivial_alpha_power_adjust = abs(np.nanmean(adjust_psd[lower_freq_index:upper_freq_index + 1]))

            return (r2, periodic_data, aperiodic_data, fooofed_data, center_frequency,
                    power, peak_width, alpha_icf, indivial_alpha_power,
                    indivial_alpha_relpower, indivial_alpha_power_adjust)

    signal_data = np.array(data['data'])
    fs = data['srate']
    # Initialize results storage
    r2_list = []
    periodic_data_list = []
    aperiodic_data_list = []
    fooofed_data_list = []
    center_frequency_list = []
    power_list = []
    peak_width_list = []
    alpha_icf_list = []
    indivial_alpha_power_list = []
    indivial_alpha_relpower_list = []
    indivial_alpha_power_adjust_list = []

    # Process each channel
    for signal in signal_data:
        iaf_fooof_values = iaf_fooof(signal, fs)
        if iaf_fooof_values:
            r2_list.append(iaf_fooof_values[0])
            periodic_data_list.append(iaf_fooof_values[1])
            aperiodic_data_list.append(iaf_fooof_values[2])
            fooofed_data_list.append(iaf_fooof_values[3])
            center_frequency_list.append(iaf_fooof_values[4])
            power_list.append(iaf_fooof_values[5])
            peak_width_list.append(iaf_fooof_values[6])
            alpha_icf_list.append(iaf_fooof_values[7])
            indivial_alpha_power_list.append(iaf_fooof_values[8])
            indivial_alpha_relpower_list.append(iaf_fooof_values[9])
            indivial_alpha_power_adjust_list.append(iaf_fooof_values[10])
        else:
            # Add NaN for missing values
            r2_list.append(np.NAN)
            periodic_data_list.append(np.NAN)
            aperiodic_data_list.append(np.NAN)
            fooofed_data_list.append(np.NAN)
            center_frequency_list.append(np.NAN)
            power_list.append(np.NAN)
            peak_width_list.append(np.NAN)
            alpha_icf_list.append(np.NAN)
            indivial_alpha_power_list.append(np.NAN)
            indivial_alpha_relpower_list.append(np.NAN)
            indivial_alpha_power_adjust_list.append(np.NAN)

    # Format results as dictionary
    iaf_fooof_result = {
        'data': None,
        'events': None,
        'srate': data['srate'],
        'nchan': data['nchan'],
        'ch_names': data['ch_names'],
        'type': 'aperiodic',
        'r2': r2_list,
        'periodic_data': periodic_data_list,
        'aperiodic_data': aperiodic_data_list,
        'fooofed_data': fooofed_data_list,
        'center_frequency': center_frequency_list,
        'power': power_list,
        'peak_width': peak_width_list,
        'alpha_icf': alpha_icf_list,
        'indivial_alpha_power': indivial_alpha_power_list,
        'indivial_alpha_relpower': indivial_alpha_relpower_list,
        'indivial_alpha_power_adjust': indivial_alpha_power_adjust_list
    }

    if is_save:
        save_file(data=iaf_fooof_result, save_path=save_path, save_filestyle=save_filestyle)

    return iaf_fooof_result


def fnirs_make_design_matrix(frame_times, conditions, onsets, duration, hrf_model="glover", drift_model="polynomial",
                             drift_order=1):
    """
    Create design matrix for fNIRS GLM analysis

    :param frame_times: Time points for each sample
    :type frame_times: np.ndarray
    :param conditions: Condition labels for each event
    :type conditions: list
    :param onsets: Onset times for each event
    :type onsets: list
    :param duration: Duration for each event
    :type duration: list
    :param hrf_model: Hemodynamic response function model (default "glover")
    :type hrf_model: str
    :param drift_model: Drift model (default "polynomial")
    :type drift_model: str
    :param drift_order: Polynomial order for drift model (default 1)
    :type drift_order: int
    :return: Design matrix
    :rtype: pd.DataFrame
    """
    events = pd.DataFrame(
        {"trial_type": conditions, "onset": onsets, "duration": duration}
    )
    design_matrix = make_first_level_design_matrix(
        frame_times,
        events,
        drift_model=drift_model,
        drift_order=drift_order,
        hrf_model=hrf_model,
    )
    return design_matrix


def fnirs_run_glm(data, design_matrix):
    """
    Run GLM analysis for fNIRS data

    :param data: fNIRS data dictionary with keys
    :type data: dict
    :param design_matrix: Design matrix from fnirs_make_design_matrix
    :type design_matrix: pd.DataFrame
    :return: GLM results per channel
    :rtype: dict
    """
    glm_results = dict()
    for chan, ch_name in zip(data['data'], data['ch_names']):
        signal = np.array(chan).reshape(1, -1)
        labels, glm_estimates = run_glm(signal.T, design_matrix)
        estimates = glm_estimates[labels[0]]
        channel_result = {
            'theta': estimates.theta,
            'MSE': estimates.MSE[0]
        }
        glm_results[ch_name] = channel_result
    return glm_results


def ecg_hrv(data):
    """
    Calculate heart rate variability (HRV) from ECG

    :param data: ECG data dictionary with keys
    :type data: dict
    :return: R-peak indices and RR-intervals
    :rtype: tuple
    """
    r_peaks, info = nk.ecg_peaks(data['data'], sampling_rate=data['srate'])
    r_peak_indices = np.where(r_peaks == 1)[0]
    rr_intervals = np.diff(r_peak_indices)
    return r_peak_indices, rr_intervals


def calculate_granger_causality_from_dict(signal, influence_signal, signal_name=None, influence_name=None, maxlag=10):
    """
    Calculate Granger causality between signal sets

    :param signal: Source signals (n_signals x n_timepoints)
    :type signal: np.ndarray
    :param influence_signal: Target signals (n_signals x n_timepoints)
    :type influence_signal: np.ndarray
    :param signal_name: Names of source signals
    :type signal_name: list, optional
    :param influence_name: Names of target signals
    :type influence_name: list, optional
    :param maxlag: Maximum time lag
    :type maxlag: int
    :return: Granger causality results
    :rtype: pd.DataFrame
    """
    # Set default names if not provided
    if signal_name is None:
        signal_name = [f'Signal_{i + 1}' for i in range(signal.shape[0])]
    if influence_name is None:
        influence_name = [f'Influence_{i + 1}' for i in range(influence_signal.shape[0])]

    results_dict = {
        'channels': [],
        'lag': [],
        'ssr_ftest(F)': [],
        'ssr_ftest(p-value)': [],
        'ssr_chi2test(F)': [],
        'ssr_chi2test(p-value)': [],
        'lrtest(F)': [],
        'lrtest(p-value)': [],
        'params_ftest(F)': [],
        'params_ftest(p-value)': [],
    }

    # Calculate Granger causality for each pair
    for i in range(signal.shape[0]):
        for j in range(influence_signal.shape[0]):
            x = np.vstack((signal[i, :], influence_signal[j, :]))
            granger_result = grangercausalitytests(x.T, maxlag=maxlag)
            for lag in granger_result.keys():
                results_dict['channels'].append(f"{signal_name[i]}-{influence_name[j]}")
                results_dict['lag'].append(lag)
                # Extract and store test results
                test_results = granger_result[lag][0]
                for test_name in ['ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest']:
                    if test_name in test_results:
                        results_dict[f'{test_name}(F)'].append(test_results[test_name][0])
                        results_dict[f'{test_name}(p-value)'].append(test_results[test_name][1])
                    else:
                        results_dict[f'{test_name}(F)'].append(np.nan)
                        results_dict[f'{test_name}(p-value)'].append(np.nan)

    results = pd.DataFrame(results_dict)
    results.to_excel("granger_results.xlsx")
    return results


def calculate_coherence_from_dict(a, b, method='pearson', a_name=None, b_name=None, is_save=True, save_folder=None):
    """
       Calculate coherence measures between signal sets

       :param a: First set of signals
       :type a: np.ndarray
       :param b: Second set of signals
       :type b: np.ndarray
       :param method: Coherence method:
           'pearson': Pearson correlation
           'coherence': Spectral coherence
           'mutual_information': Mutual information
           'phase_sync': Phase synchrony
       :type method: str
       :param a_name: Names for signals in set A
       :type a_name: list, optional
       :param b_name: Names for signals in set B
       :type b_name: list, optional
       :param is_save: Save results flag
       :type is_save: bool
       :param save_folder: Output folder
       :type save_folder: str, optional
       :return: Coherence matrix
       :rtype: pd.DataFrame
       """
    def discretize_signal(signal, num_bins):
        """Discretize signal for mutual information calculation"""
        hist, bin_edges = np.histogram(signal, bins=num_bins)
        discrete_signal = np.digitize(signal, bin_edges[:-1])
        return discrete_signal

    def calculate_phase_sync(signal1, signal2):
        """Calculate phase synchrony"""
        phase_diff = np.angle(signal1) - np.angle(signal2)
        phase_diff_wrapped = np.angle(np.exp(1j * phase_diff))
        phase_sync = np.abs(np.mean(np.exp(1j * phase_diff_wrapped)))
        return phase_sync

    if a_name is None:
        a_name = [f'A_{i + 1}' for i in range(a.shape[0])]
    if b_name is None:
        b_name = [f'B_{i + 1}' for i in range(b.shape[0])]

    if method == 'pearson':
        results = np.zeros((a.shape[0], b.shape[0]))
        for i in range(a.shape[0]):
            for j in range(b.shape[0]):
                results[i, j], _ = pearsonr(a[i, :], b[j, :])
        filename = 'pearson_results.xlsx'
    elif method == 'coherence':
        results = np.zeros((a.shape[0], b.shape[0]))
        for i in range(a.shape[0]):
            for j in range(b.shape[0]):
                results[i, j] = np.mean(coherence(a[i, :], b[j, :])[1])
        filename = 'coherence_results.xlsx'
    elif method == 'mutual_information':
        results = np.zeros((a.shape[0], b.shape[0]))
        for i in range(a.shape[0]):
            for j in range(b.shape[0]):
                a_dis = discretize_signal(a[i, :], 1000)
                b_dis = discretize_signal(b[j, :], 1000)
                results[i, j] = mutual_info_score(a_dis, b_dis)
        filename = 'mutual_information_results.xlsx'
    elif method == 'phase_sync':
        results = np.zeros((a.shape[0], b.shape[0]))
        for i in range(a.shape[0]):
            for j in range(b.shape[0]):
                results[i, j] = calculate_phase_sync(a[i, :], b[j, :])
        filename = 'phase_sync_results.xlsx'

    df = pd.DataFrame(results, index=a_name, columns=b_name).style

    if is_save:
        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            filename = os.path.join(save_folder, filename)
        df.to_excel(filename, engine='openpyxl')
