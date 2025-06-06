# -*- coding: utf-8 -*-
# Neurovascular Coupling Analysis Module
# Provides functions for quantifying coupling between EEG and fNIRS signals

import json
import os
import zipfile
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from nilearn.glm.first_level import spm_hrf

from BrainFusion.io.File_IO import save_json, save_metadata
from BrainFusion.utils.normalize import slice_by_minlength, normalize, convert_to_serializable
from BrainFusion.utils.signals import resample_signal, compute_psd, compute_mean

# Configure matplotlib backend for interactive plotting
matplotlib.use('QtAgg')


def compute_neurovascular_coupling(eeg_signal, eeg_srate, fnirs_signal, fnirs_srate,
                                   window_size, eeg_processing_method, hrf_tr,
                                   hrf_oversampling, hrf_time_length, display_plots=True,
                                   fnirs_resample_rate=None):
    """
    Compute neurovascular coupling coefficient between EEG and fNIRS signals

    :param eeg_signal: Raw EEG timeseries data
    :type eeg_signal: numpy.ndarray
    :param eeg_srate: EEG sampling rate (Hz)
    :type eeg_srate: float
    :param fnirs_signal: Raw fNIRS timeseries data
    :type fnirs_signal: numpy.ndarray
    :param fnirs_srate: fNIRS sampling rate (Hz)
    :type fnirs_srate: float
    :param window_size: Processing window size in samples
    :type window_size: int
    :param eeg_processing_method: EEG processing technique ('avg_psd', 'resample_raw', 'avg_raw')
    :type eeg_processing_method: str
    :param hrf_tr: HRF repetition time parameter
    :type hrf_tr: float
    :param hrf_oversampling: HRF oversampling factor
    :type hrf_oversampling: int
    :param hrf_time_length: HRF time length parameter
    :type hrf_time_length: float
    :param display_plots: Flag to show visualization plots
    :type display_plots: bool
    :param fnirs_resample_rate: Target rate for fNIRS resampling
    :type fnirs_resample_rate: float, optional
    :return: Convolved EEG signal and Pearson correlation coefficient
    :rtype: tuple(numpy.ndarray, float)
    """
    # Create local copy of EEG signal
    processed_eeg = eeg_signal.copy()

    # Process EEG based on specified method
    if eeg_processing_method == 'avg_psd':
        processed_eeg = np.array([
            compute_psd(
                processed_eeg[i:i + window_size],
                fs=eeg_srate,
                nperseg=window_size,
                fl=0,
                fh=int(eeg_srate / 2)
            )
            for i in range(0, len(processed_eeg) - window_size + 1, window_size)
        ])
    elif eeg_processing_method == 'resample_raw':
        processed_eeg = resample_signal(processed_eeg, eeg_srate, eeg_srate / window_size)
    elif eeg_processing_method == 'avg_raw':
        processed_eeg = compute_mean(processed_eeg, window_size)

    # Normalize processed EEG signal
    processed_eeg = normalize(processed_eeg)

    # Generate hemodynamic response function
    hrf = spm_hrf(hrf_tr, oversampling=hrf_oversampling, time_length=hrf_time_length)

    # Convolve EEG with HRF
    convolved_signal = np.convolve(processed_eeg, hrf, mode='full')[:len(processed_eeg)]

    # Process and normalize fNIRS signal
    processed_fnirs = fnirs_signal.copy()
    if fnirs_resample_rate:
        processed_fnirs = resample_signal(processed_fnirs, int(fnirs_srate), int(fnirs_resample_rate))

    processed_fnirs_norm = normalize(processed_fnirs)
    convolved_signal_norm = normalize(convolved_signal)[:len(processed_fnirs)]

    # Synchronize signal lengths
    convolved_signal_norm, processed_fnirs_norm = slice_by_minlength(
        convolved_signal_norm, processed_fnirs_norm
    )

    # Check for invalid values
    if (np.isnan(convolved_signal_norm).any() or
            np.isnan(processed_fnirs_norm).any() or
            np.isinf(convolved_signal_norm).any() or
            np.isinf(processed_fnirs_norm).any()):
        print("Skipping channel due to invalid signal values")
        return None, None

    # Compute Pearson correlation
    correlation, _ = pearsonr(convolved_signal_norm, processed_fnirs_norm)

    # Visualize results if requested
    if display_plots:
        plt.figure(figsize=(12, 6))
        plt.title("Neurovascular Coupling Analysis")
        plt.plot(convolved_signal_norm, label='EEG-derived Hemodynamic Response')
        plt.plot(processed_fnirs_norm, label='Actual fNIRS Signal')
        plt.plot(processed_eeg, label='Processed EEG Signal')
        plt.xlabel("Samples")
        plt.ylabel("Normalized Amplitude")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return convolved_signal, correlation


def compute_neurovascular_coupling_by_dict(eeg_dict, fnirs_dict, window_size, eeg_processing_method,
                                           hrf_tr, hrf_oversampling, hrf_time_length, display_plots=False,
                                           fnirs_resample_rate=None, save_results=False, output_path=None,
                                           file_type='.xlsx'):
    """
    Compute neurovascular coupling for all channel combinations

    :param eeg_dict: EEG data dictionary with 'data', 'ch_names', and 'srate'
    :type eeg_dict: dict
    :param fnirs_dict: fNIRS data dictionary with 'data', 'ch_names', and 'srate'
    :type fnirs_dict: dict
    :param window_size: Processing window size in samples
    :type window_size: int
    :param eeg_processing_method: EEG processing method selector
    :type eeg_processing_method: str
    :param hrf_tr: HRF repetition time parameter
    :type hrf_tr: float
    :param hrf_oversampling: HRF oversampling factor
    :type hrf_oversampling: int
    :param hrf_time_length: HRF time length parameter
    :type hrf_time_length: float
    :param display_plots: Show individual channel plots
    :type display_plots: bool
    :param fnirs_resample_rate: fNIRS target resampling rate
    :type fnirs_resample_rate: float, optional
    :param save_results: Flag to enable result saving
    :type save_results: bool
    :param output_path: Directory for saving analysis outputs
    :type output_path: str, optional
    :param file_type: Output file extension ('.xlsx' or '.csv')
    :type file_type: str
    :return: DataFrame with analysis results
    :rtype: pandas.DataFrame
    """
    # Extract data from dictionaries
    eeg_data = eeg_dict['data']
    eeg_channels = eeg_dict['ch_names']
    eeg_srate = eeg_dict['srate']
    fnirs_data = fnirs_dict['data']
    fnirs_srate = fnirs_dict['srate']
    fnirs_channels = fnirs_dict['ch_names']

    results = []

    # Calculate coupling for all channel combinations
    for eeg_idx in range(eeg_data.shape[0]):
        for fnirs_idx in range(fnirs_data.shape[0]):
            eeg_ts = eeg_data[eeg_idx]
            fnirs_ts = fnirs_data[fnirs_idx]

            _, correlation = compute_neurovascular_coupling(
                eeg_ts, eeg_srate,
                fnirs_ts, fnirs_srate,
                window_size, eeg_processing_method,
                hrf_tr, hrf_oversampling, hrf_time_length,
                display_plots, fnirs_resample_rate
            )

            # Save valid results
            if correlation is not None:
                results.append((
                    eeg_channels[eeg_idx],
                    fnirs_channels[fnirs_idx],
                    correlation
                ))

    # Create results DataFrame
    results_df = pd.DataFrame(
        results,
        columns=['EEG Channel', 'fNIRS Channel', 'Pearson Correlation']
    )

    # Save results if requested
    if save_results and output_path:
        # Create metadata dictionaries
        eeg_metadata = {k: v for k, v in eeg_dict.items() if k not in ['data', 'events', 'loc']}
        fnirs_metadata = {k: v for k, v in fnirs_dict.items() if k not in ['data', 'events', 'loc']}

        # Assemble analysis metadata
        analysis_metadata = {
            'eeg_metadata': eeg_metadata,
            'fnirs_metadata': fnirs_metadata,
            'analysis_type': 'neurovascular_coupling',
            'processing_parameters': {
                'window_size': window_size,
                'eeg_processing_method': eeg_processing_method,
                'hrf_tr': hrf_tr,
                'hrf_oversampling': hrf_oversampling,
                'hrf_time_length': hrf_time_length,
                'fnirs_resample_rate': fnirs_resample_rate,
                'timestamp': datetime.now().isoformat()
            }
        }

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Create spreadsheet file path
        spreadsheet_filename = os.path.splitext(output_path)[0] + file_type

        # Save results to spreadsheet
        if file_type == '.xlsx':
            results_df.to_excel(spreadsheet_filename, sheet_name='Coupling Results', index=False)
        else:  # Default to CSV
            results_df.to_csv(spreadsheet_filename, index=False)

        # Create metadata file
        metadata_filename = os.path.splitext(output_path)[0] + '_metadata.json'
        save_metadata(analysis_metadata, metadata_filename)

        # Create ZIP archive containing both files
        with zipfile.ZipFile(output_path, 'w') as archive:
            archive.write(spreadsheet_filename, os.path.basename(spreadsheet_filename))
            archive.write(metadata_filename, os.path.basename(metadata_filename))

        # Remove temporary files
        os.remove(spreadsheet_filename)
        os.remove(metadata_filename)

        print(f"Results saved to: {output_path}")

    return results_df


if __name__ == "__main__":
    # Sample EEG data structure
    sample_eeg = {
        'data': np.random.randn(3, 10000),  # 3 EEG channels, 10000 samples each
        'srate': 250,  # 250 Hz sampling rate
        'ch_names': ['Fp1', 'Fp2', 'F3']
    }

    # Sample fNIRS data structure
    sample_fnirs = {
        'data': np.random.randn(2, 1100),  # 2 fNIRS channels, 1100 samples each
        'srate': 10,  # 10 Hz sampling rate
        'ch_names': ['S1-D1', 'S2-D2']
    }

    # Analysis parameters
    analysis_params = {
        'window_size': 50,  # EEG processing window (in samples)
        'eeg_processing_method': 'avg_psd',  # 'avg_psd', 'resample_raw', or 'avg_raw'
        'hrf_tr': 1.0,  # HRF repetition time
        'hrf_oversampling': 1,  # HRF oversampling factor
        'hrf_time_length': 32.0,  # HRF time length
        'display_plots': False,  # Disable individual channel plots
        'fnirs_resample_rate': None,  # Optional fNIRS resampling rate
        'save_results': True,  # Save analysis outputs
        'output_path': 'results/neurovascular_coupling.zip',  # Output location
        'file_type': '.xlsx'  # Output file format
    }

    # Perform analysis
    results_df = compute_neurovascular_coupling_by_dict(
        sample_eeg, sample_fnirs, **analysis_params
    )

    # Display results
    print("\nNeurovascular Coupling Results:")
    print(results_df)