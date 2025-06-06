"""
SNIRF and BDF Conversion Utilities

Provides functions for converting neuroscience datasets to various formats
including BDF, SNIRF, and BrainFusion's internal format.
Compatible with Sphinx documentation generator.
"""

import os.path
import numpy as np
import pandas as pd
import scipy.io
import mne
import mne_nirs
from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import run_glm
from nilearn.plotting import plot_design_matrix

from BrainFusion.io.File_IO import save_mat, read_mat, create_standard_bdf_file
from BrainFusion.io.snirf_io import create_snirf_file


def remove_dtype(value):
    """
    Convert NumPy arrays to Python lists for serialization

    :param value: Input value to process
    :return: Value with arrays converted to lists
    :rtype: list or original type
    """
    if isinstance(value, np.ndarray):
        return value.tolist()  # Convert NumPy arrays to Python lists
    return value  # Return unchanged if not array


def read_tu_berlin_bci_eeg_data(folder_path):
    """
    Load TU Berlin BCI EEG dataset subject folder

    :param folder_path: Path to subject folder
    :type folder_path: str
    :return: EEG data organized in dictionary
    :rtype: dict
    """
    cnt_file = os.path.join(folder_path, 'cnt.mat')
    mrk_file = os.path.join(folder_path, 'mrk.mat')
    mnt_file = os.path.join(folder_path, 'mnt.mat')

    cnt_data = scipy.io.loadmat(cnt_file, struct_as_record=False, squeeze_me=True)
    mrk_data = scipy.io.loadmat(mrk_file, struct_as_record=False, squeeze_me=True)
    mnt_data = scipy.io.loadmat(mnt_file, struct_as_record=False, squeeze_me=True)

    cnt = cnt_data['cnt']
    mrk = mrk_data['mrk']
    mnt = mnt_data['mnt']

    combined_data = {}

    for i, (cnt_struct, mrk_struct) in enumerate(zip(cnt, mrk)):
        struct_key = f"struct_{i + 1}"
        eeg_dict = {
            'data': remove_dtype(cnt_struct.x.T / 1000000),
            'num_channels': len(remove_dtype(cnt_struct.clab)),
            'channel_names': remove_dtype(cnt_struct.clab),
            'sampling_rate': remove_dtype(cnt_struct.fs),
            'title': remove_dtype(cnt_struct.title),
            'time': remove_dtype(cnt_struct.T),
            'units': remove_dtype(cnt_struct.yUnit),
            'signal_type': 'eeg',
            'montage': 'standard_1005',
        }

        # Process events
        event_times = remove_dtype(mrk_struct.time)
        event_ids = remove_dtype(mrk_struct.event.desc)
        events = [[time, 0, event_id] for time, event_id in zip(event_times, event_ids)]
        eeg_dict['events'] = events

        # Add electrode positions
        eeg_dict['locations'] = {
            'x': remove_dtype(mnt.x),
            'y': remove_dtype(mnt.y),
            'positions_3d': remove_dtype(mnt.positions_3d),
            'channel_names': remove_dtype(mnt.clab),
        }

        combined_data[struct_key] = eeg_dict

    return combined_data


def convert_tu_berlin_eeg_to_brainfusion(source_path, output_folder):
    """
    Convert TU Berlin EEG dataset to BrainFusion format

    :param source_path: Path to source data
    :type source_path: str
    :param output_folder: Output folder
    :type output_folder: str
    """
    data_dict = read_tu_berlin_bci_eeg_data(source_path)
    os.makedirs(output_folder, exist_ok=True)

    for key, value in data_dict.items():
        struct_num = int(key.split('_')[1])
        if struct_num % 2 == 1:
            session_type = 'MI'
            session_num = struct_num // 2 + 1
        else:
            session_type = 'MA'
            session_num = struct_num // 2

        file_name = f"{session_type}_{session_num}.mat"
        save_mat(value, os.path.join(output_folder, file_name))
        print(f'Converted session: {file_name}')


def convert_tu_berlin_nirs_to_brainfusion(source_path, output_folder):
    """
    Convert TU Berlin NIRS dataset to BrainFusion format

    :param source_path: Path to source data
    :type source_path: str
    :param output_folder: Output folder
    :type output_folder: str
    """
    data_dict = read_tu_berlin_bci_nirs_data(source_path)
    os.makedirs(output_folder, exist_ok=True)

    for key, value in data_dict.items():
        struct_num = int(key.split('_')[1])
        if struct_num % 2 == 1:
            session_type = 'MI'
            session_num = struct_num // 2 + 1
        else:
            session_type = 'MA'
            session_num = struct_num // 2

        file_name = f"{session_type}_{session_num}.mat"
        save_mat(value, os.path.join(output_folder, file_name))
        print(f'Converted session: {file_name}')


def convert_tu_berlin_to_snirf(mat_path, output_path):
    """
    Convert TU Berlin dataset to SNIRF format

    :param mat_path: Path to MATLAB file
    :type mat_path: str
    :param output_path: Output SNIRF file path
    :type output_path: str
    """
    data_dict = read_mat(mat_path)
    data_dict = filter_optode_positions(data_dict)
    data_time_series = np.array(data_dict['data']).T
    time_points = generate_time_series(np.array(data_dict['data']).shape[1], data_dict['sampling_rate'])

    # Create measurement lists
    measurement_lists = []
    cnt = 0
    for pair in data_dict['source_detector_pairs']:
        for j in range(2):
            measurement_lists.append({
                'sourceIndex': pair[0],
                'detectorIndex': pair[1],
                'wavelengthIndex': j + 1,
                'dataType': 1,
                'dataTypeIndex': 1,
            })
            cnt += 1

    # Process events
    for event in data_dict['events']:
        event[0] = event[0] / 1000.0  # Convert to seconds

    stim_mi_left = {'name': 'MI_Left', 'data': []}
    stim_mi_right = {'name': 'MI_Right', 'data': []}

    for event in data_dict['events']:
        if event[2] == 1:
            stim_mi_left['data'].append(event)
        elif event[2] == 2:
            stim_mi_right['data'].append(event)

    stim_list = [stim_mi_left, stim_mi_right]

    # Create SNIRF file
    create_snirf_file(
        filename=output_path,
        data_time_series=data_time_series,
        time_points=time_points,
        measurement_lists=measurement_lists,
        source_pos_3d=data_dict['source_positions_3d'],
        detector_pos_3d=data_dict['detector_positions_3d'],
        wavelengths=data_dict['wavelengths'],
        landmark_labels=data_dict['landmark_labels'],
        stim_lists=stim_list
    )
    print(f"Created SNIRF file: {output_path}")


def convert_dataset_to_brainfusion_nirs(data_root, output_root):
    """
    Convert entire TU Berlin NIRS dataset to BrainFusion format

    :param data_root: Root directory of dataset
    :type data_root: str
    :param output_root: Output root directory
    :type output_root: str
    """
    nirs_folder = os.path.join(data_root, 'NIRS_01-29')
    os.makedirs(output_root, exist_ok=True)

    for subject in os.listdir(nirs_folder):
        subject_path = os.path.join(nirs_folder, subject)
        subject_output = os.path.join(output_root, subject)
        convert_tu_berlin_nirs_to_brainfusion(subject_path, subject_output)


def convert_dataset_to_brainfusion_eeg(data_root, output_root):
    """
    Convert entire TU Berlin EEG dataset to BrainFusion format

    :param data_root: Root directory of dataset
    :type data_root: str
    :param output_root: Output root directory
    :type output_root: str
    """
    eeg_folders = [
        'EEG_01-05', 'EEG_06-10', 'EEG_11-15',
        'EEG_16-20', 'EEG_21-25', 'EEG_26-29'
    ]

    for eeg_folder in eeg_folders:
        folder_path = os.path.join(data_root, eeg_folder)
        for subject in os.listdir(folder_path):
            subject_path = os.path.join(folder_path, subject, 'with_ocular_artifact')
            subject_output = os.path.join(output_root, subject)
            os.makedirs(subject_output, exist_ok=True)
            convert_tu_berlin_eeg_to_brainfusion(subject_path, subject_output)


def convert_dataset_to_snirf(data_root, output_root):
    """
    Convert entire TU Berlin dataset to SNIRF format

    :param data_root: Root directory of dataset
    :type data_root: str
    :param output_root: Output root directory
    :type output_root: str
    """
    for subject in os.listdir(data_root):
        subject_path = os.path.join(data_root, subject)
        os.makedirs(os.path.join(output_root, subject), exist_ok=True)

        for session in os.listdir(subject_path):
            if 'MI' in session:
                input_path = os.path.join(subject_path, session)
                output_path = os.path.join(
                    output_root, subject,
                    session.replace('.mat', '.snirf')
                )
                convert_tu_berlin_to_snirf(input_path, output_path)


def convert_dataset_to_bdf(data_root, output_root):
    """
    Convert entire TU Berlin dataset to BDF format

    :param data_root: Root directory of dataset
    :type data_root: str
    :param output_root: Output root directory
    :type output_root: str
    """
    for subject in os.listdir(data_root):
        subject_path = os.path.join(data_root, subject)
        os.makedirs(os.path.join(output_root, subject), exist_ok=True)

        for session in os.listdir(subject_path):
            if 'MI' in session:
                input_path = os.path.join(subject_path, session)
                output_path = os.path.join(
                    output_root, subject,
                    session.replace('.mat', '.bdf')
                )
                convert_tu_berlin_to_bdf(input_path, output_path)


def convert_tu_berlin_to_bdf(mat_path, output_path):
    """
    Convert TU Berlin data to BDF format

    :param mat_path: MATLAB file path
    :type mat_path: str
    :param output_path: Output BDF file path
    :type output_path: str
    """
    data_dict = read_mat(mat_path)

    create_standard_bdf_file(
        file_name=output_path,
        num_channels=len(data_dict['channel_names']),
        signals=np.array(data_dict['data']) * 1000000,
        channel_names=data_dict['channel_names'],
        sampling_frequency=data_dict['sampling_rate'],
        annotations=data_dict['events']
    )


def read_tu_berlin_bci_nirs_data(folder_path):
    """
    Load TU Berlin BCI NIRS dataset subject folder

    :param folder_path: Path to subject folder
    :type folder_path: str
    :return: NIRS data organized in dictionary
    :rtype: dict
    """
    cnt_file = os.path.join(folder_path, 'cnt.mat')
    mrk_file = os.path.join(folder_path, 'mrk.mat')
    mnt_file = os.path.join(folder_path, 'mnt.mat')

    cnt_data = scipy.io.loadmat(cnt_file, struct_as_record=False, squeeze_me=True)
    mrk_data = scipy.io.loadmat(mrk_file, struct_as_record=False, squeeze_me=True)
    mnt_data = scipy.io.loadmat(mnt_file, struct_as_record=False, squeeze_me=True)

    cnt = cnt_data['cnt']
    mrk = mrk_data['mrk']
    mnt = mnt_data['mnt']

    nirs_data = {}

    for i, (cnt_struct, mrk_struct) in enumerate(zip(cnt, mrk)):
        struct_key = f"struct_{i + 1}"
        nirs_dict = {
            'channel_names': remove_dtype(cnt_struct.clab),
            'sampling_rate': remove_dtype(cnt_struct.fs),
            'title': remove_dtype(cnt_struct.title),
            'data': remove_dtype(cnt_struct.x.T),
            'wavelengths': remove_dtype(cnt_struct.wavelengths),
            'signal_type': 'fnirs',
            'num_channels': len(remove_dtype(cnt_struct.clab)),
            'montage': None,
        }

        # Process events
        event_times = remove_dtype(mrk_struct.time)
        event_ids = remove_dtype(mrk_struct.event.desc)
        events = [[time, 0, event_id] for time, event_id in zip(event_times, event_ids)]
        nirs_dict['events'] = events

        # Process positions
        nirs_dict.update({
            'source_positions_2d': [
                [x, y] for x, y in
                zip(remove_dtype(mnt.source.x), remove_dtype(mnt.source.y))
            ],
            'detector_positions_2d': [
                [x, y] for x, y in
                zip(remove_dtype(mnt.detector.x), remove_dtype(mnt.detector.y))
            ],
            'source_positions_3d': remove_dtype(mnt.source.positions_3d.T),
            'detector_positions_3d': remove_dtype(mnt.detector.positions_3d.T),
            'source_labels': remove_dtype(mnt.source.clab),
            'detector_labels': remove_dtype(mnt.detector.clab),
            'landmark_positions_2d': [
                [x, y] for x, y in
                zip(remove_dtype(mnt.x), remove_dtype(mnt.y))
            ],
            'landmark_positions_3d': remove_dtype(mnt.positions_3d.T),
            'landmark_labels': remove_dtype(mnt.clab),
            'source_detector_pairs': remove_dtype(mnt.sd),
        })

        nirs_dict = interleave_channels(nirs_dict)
        nirs_data[struct_key] = nirs_dict

    return nirs_data


def interleave_channels(data):
    """
    Interleave fNIRS channel data by wavelength

    :param data: Input data dictionary
    :return: Data with interleaved channels
    :rtype: dict
    """
    num_channels = len(data['channel_names'])
    if num_channels % 2 != 0:
        raise ValueError("Number of channels must be even")

    half_size = num_channels // 2
    interleaved_names = []
    interleaved_data = []

    for i in range(half_size):
        # Append first wavelength channel
        interleaved_names.append(data['channel_names'][i])
        interleaved_data.append(data['data'][i])

        # Append second wavelength channel
        interleaved_names.append(data['channel_names'][i + half_size])
        interleaved_data.append(data['data'][i + half_size])

    data['channel_names'] = interleaved_names
    data['data'] = interleaved_data

    return data


def generate_time_series(num_samples, sampling_rate):
    """
    Generate time vector for signals

    :param num_samples: Number of samples
    :type num_samples: int
    :param sampling_rate: Sampling frequency in Hz
    :type sampling_rate: float
    :return: Time vector in seconds
    :rtype: numpy.ndarray
    """
    return np.linspace(0, (num_samples - 1) / sampling_rate, num_samples)


def filter_optode_positions(data):
    """
    Filter invalid optode positions marked with '-'

    :param data: Input data dictionary
    :return: Data with filtered positions
    :rtype: dict
    """
    # Identify invalid sources
    source_mask = ['-' in label for label in data['source_labels']]
    detector_mask = ['-' in label for label in data['detector_labels']]

    # Adjust source-detector pairs
    sd_pairs = data['source_detector_pairs']
    offset = 0
    for i, is_invalid in enumerate(source_mask):
        if is_invalid:
            for pair in sd_pairs:
                if pair[0] > i - offset:
                    pair[0] -= 1
            offset += 1

    # Filter positions
    data['source_positions_3d'] = np.array([
        pos for pos, is_invalid in zip(data['source_positions_3d'], source_mask)
        if not is_invalid
    ]) * 10

    data['detector_positions_3d'] = np.array([
        pos for pos, is_invalid in zip(data['detector_positions_3d'], detector_mask)
        if not is_invalid
    ]) * 10

    # Apply coordinate adjustments
    data['source_positions_3d'][:, 2] += 5
    data['detector_positions_3d'][:, 2] += 5
    data['source_positions_3d'][:, 1] += 1
    data['detector_positions_3d'][:, 1] += 1

    return data