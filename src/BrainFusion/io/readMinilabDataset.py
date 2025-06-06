"""
SNIRF and BDF File Processing Utilities

Provides functions for reading, writing, and converting biomedical data files.
"""

import os.path
from tkinter import filedialog, Tk
import numpy as np
import pandas as pd
import pyedflib
import snirf

from scipy.io import loadmat
from BrainFusion.io.File_IO import create_standard_bdf_file, read_neuracle_bdf, save_file, dict_to_bdf, dict_to_snirf
from BrainFusion.io.snirf_io import create_snirf_file


def combine_data_and_evt(input_folder, output_file):
    """
    Combine BDF data and event files.

    :param input_folder: Input directory path
    :type input_folder: str
    :param output_file: Output file path
    :type output_file: str
    """
    data = read_neuracle_bdf(path=(os.path.join(input_folder, 'data.bdf'),
                                   os.path.join(input_folder, 'evt.bdf')))
    n_channels = len(data['channel_names'])
    data['file_info'] = {
        'PhysicalMinimum': [-37500] * n_channels,
        'PhysicalMaximum': [37500] * n_channels,
        'DigitalMinimum': [-838860] * n_channels,
        'DigitalMaximum': [838860] * n_channels,
    }
    dict_to_bdf(data_dict=data, output_path=output_file)


def save_events_to_excel_from_bdf(input_file, output_file):
    """
    Save BDF events to Excel.

    :param input_file: BDF file path
    :type input_file: str
    :param output_file: Excel file path
    :type output_file: str
    """
    data = read_minilab_bdf(input_file)
    events = data['events']
    save_events_to_excel(events=events, output_path=output_file)
    print(f'Events from {input_file} saved to {output_file}')


def save_events_to_excel_from_snirf(input_file, output_file):
    """
    Save SNIRF events to Excel.

    :param input_file: SNIRF file path
    :type input_file: str
    :param output_file: Excel file path
    :type output_file: str
    """
    data = read_minilab_snirf(input_file)
    events = data['events']
    save_events_to_excel(events=events, output_path=output_file)
    print(f'Events from {input_file} saved to {output_file}')


def add_events_to_bdf(input_bdf_file, output_bdf_file, events_file):
    """
    Add events to BDF file.

    :param input_bdf_file: Input BDF file
    :type input_bdf_file: str
    :param output_bdf_file: Output BDF file
    :type output_bdf_file: str
    :param events_file: Events Excel file
    :type events_file: str
    """
    input_data = read_minilab_bdf(input_bdf_file)
    events = load_events_from_excel(events_file)
    input_data['events'] = events.copy()
    dict_to_bdf(input_data, output_path=output_bdf_file)
    print(f'Events from {events_file} added to {input_bdf_file}')


def add_events_to_snirf(input_snirf_file, output_snirf_file, events_file):
    """
    Add events to SNIRF file.

    :param input_snirf_file: Input SNIRF file
    :type input_snirf_file: str
    :param output_snirf_file: Output SNIRF file
    :type output_snirf_file: str
    :param events_file: Events Excel file
    :type events_file: str
    """
    input_data = read_minilab_snirf(input_snirf_file)
    events = load_events_from_excel(events_file)
    input_data['events'] = events.copy()
    dict_to_snirf(input_data, output_path=output_snirf_file)
    print(f'Events from {events_file} added to {input_snirf_file}')


def read_minilab_bdf(path, signal_type='eeg', montage=''):
    """
    Read MiniLab BDF file.

    :param path: BDF file path
    :type path: str
    :param signal_type: Signal type (default 'eeg')
    :type signal_type: str
    :param montage: Electrode montage
    :type montage: str
    :return: Parsed BDF data
    :rtype: dict
    """
    bdf_file = pyedflib.EdfReader(path)
    n_channels = bdf_file.signals_in_file
    data = []

    for i in range(n_channels):
        signal = bdf_file.readSignal(i, digital=False) * 0.000001
        data.append(signal)

    # Parse events
    events = np.array(bdf_file.readAnnotations()).T.tolist()
    processed_events = []
    for event in events:
        time, duration, label = event
        label = str(int(float(label)))
        time = float(time)
        duration = float(duration)
        processed_events.append([time, duration, label])

    result = {
        'data': data,
        'sampling_rate': bdf_file.getSampleFrequencies()[0],
        'events': processed_events,
        'num_channels': n_channels,
        'channel_names': [label.replace('.', '') for label in bdf_file.getSignalLabels()],
        'signal_type': signal_type,
        'montage': montage,
        'file_info': {
            'PhysicalMaximum': bdf_file.getPhysicalMaximum(),
            'PhysicalMinimum': bdf_file.getPhysicalMinimum(),
            'DigitalMaximum': bdf_file.getDigitalMaximum(),
            'DigitalMinimum': bdf_file.getDigitalMinimum()
        }
    }

    bdf_file.close()
    return result


def read_minilab_snirf(path=None, signal_type='fnirs'):
    """
    Read MiniLab SNIRF file.

    :param path: SNIRF file path
    :type path: str
    :param signal_type: Signal type (default 'fnirs')
    :type signal_type: str
    :return: Parsed SNIRF data
    :rtype: dict
    """
    snirf_file = snirf.loadSnirf(path)
    nirs_data = snirf_file.nirs[0]

    # Time-series data
    data_dict = {
        'data': nirs_data.data[0].dataTimeSeries,
        'time': nirs_data.data[0].time,
        'signal_type': signal_type
    }

    # Event markers
    events = []
    if len(nirs_data.stim) > 0:
        for stim in nirs_data.stim:
            if stim.data is not None and len(stim.data) > 0:
                for event in stim.data:
                    time, duration, label = event
                    time = float(time)
                    duration = float(duration)
                    label = str(int(float(label)))
                    events.append([time, duration, label])

    if events:
        events.sort(key=lambda x: x[0])
        data_dict['events'] = events
    else:
        data_dict['events'] = None

    # Channel information
    data_dict['num_channels'] = data_dict['data'].shape[1]
    data_dict['channel_names'] = nirs_data.probe.landmarkLabels

    # Optode positions
    data_dict['locations'] = {
        'source_positions': nirs_data.probe.sourcePos3D,
        'detector_positions': nirs_data.probe.detectorPos3D,
        'landmark_positions': nirs_data.probe.landmarkPos3D
    }

    # Source-detector pairs
    sd_pairs = []
    for ch_name in data_dict['channel_names'][:data_dict['num_channels'] // 2]:
        parts = ch_name.split(" ")[0]
        source, detector = parts.split("_")
        source_idx = int(source[1:])
        detector_idx = int(detector[1:])
        sd_pairs.append((source_idx, detector_idx))

    data_dict['source_detector_pairs'] = sd_pairs
    data_dict['wavelengths'] = nirs_data.probe.wavelengths

    return data_dict


def add_pos_to_minilab_snirf(input_snirf_file, output_snirf_file, pos_file, events=None):
    """
    Add positions to SNIRF file.

    :param input_snirf_file: Input SNIRF file
    :type input_snirf_file: str
    :param output_snirf_file: Output SNIRF file
    :type output_snirf_file: str
    :param pos_file: Position data file
    :type pos_file: str
    :param events: Events to add (optional)
    :type events: list
    """
    snirf_file = snirf.loadSnirf(input_snirf_file)
    data = pd.read_excel(pos_file, header=None)

    sources = data[data[0].str.startswith('S')].reset_index(drop=True)
    detectors = data[data[0].str.startswith('D')].reset_index(drop=True)
    channels = data[data[0].str.startswith('CH')].reset_index(drop=True)

    source_positions = np.array([row[1:4].values for _, row in sources.iterrows()]) / 8
    detector_positions = np.array([row[1:4].values for _, row in detectors.iterrows()]) / 8

    # Apply adjustments
    source_positions[:, 2] += 5
    detector_positions[:, 2] += 5
    source_positions[:, 1] += 3
    detector_positions[:, 1] += 3

    # Landmarks
    landmarks = []
    labels = []
    wavelengths = snirf_file.nirs[0].probe.wavelengths
    for wavelength in wavelengths:
        for _, row in channels.iterrows():
            name = row[0]
            pos = row[1:4].values / 8
            pos[2] += 5
            pos[1] += 3
            parts = name.split('(')[1].split(')')[0].replace('-', '_')
            labels.append(f"{parts} {int(wavelength)}")
            landmarks.append(pos)

    # Events
    stim_data = []
    if events:
        for event in events:
            onset, duration, label = event
            stim_data.append({
                'name': str(int(float(label))),
                'data': [[onset, duration, float(label)]]
            })

    create_snirf_file(
        filename=output_snirf_file,
        data_time_series=snirf_file.nirs[0].data[0].dataTimeSeries,
        time_points=snirf_file.nirs[0].data[0].time,
        measurement_lists=[
            {
                'sourceIndex': ml.sourceIndex,
                'detectorIndex': ml.detectorIndex,
                'wavelengthIndex': ml.wavelengthIndex,
                'dataType': ml.dataType,
                'dataTypeIndex': ml.dataTypeIndex,
            }
            for ml in snirf_file.nirs[0].data[0].measurementList
        ],
        source_pos_3d=source_positions.astype(np.float64),
        detector_pos_3d=detector_positions.ast(np.float64),
        wavelengths=wavelengths,
        landmark_labels=labels,
        landmark_pos_3d=np.array(landmarks).astype(np.float64),
        stim_lists=stim_data
    )
    print(f"SNIRF file updated: {output_snirf_file}")


def add_event_to_minilab_bdf(data_dict, output_file, events=None):
    """
    Add events to BDF file.

    :param data_dict: Data dictionary
    :type data_dict: dict
    :param output_file: Output file path
    :type output_file: str
    :param events: Events to add
    :type events: list
    """
    num_channels = len(data_dict['channel_names'])
    signals = np.array(data_dict['data']) * 1000000

    if events is None:
        annotations = data_dict['events']
    else:
        annotations = events

    for ann in annotations:
        ann[0] = float(ann[0])
        ann[1] = float(ann[1])

    create_standard_bdf_file(
        file_name=output_file,
        num_channels=num_channels,
        signals=signals,
        channel_names=data_dict['channel_names'],
        physical_mins=[-37500.0] * num_channels,
        physical_maxs=[37500.0] * num_channels,
        digital_mins=[-838860.0] * num_channels,
        digital_maxs=[838860.0] * num_channels,
        sampling_frequency=data_dict['sampling_rate'],
        annotations=annotations
    )


def load_events_from_excel(input_path):
    """
    Load events from Excel.

    :param input_path: Excel file path
    :type input_path: str
    :return: List of events
    :rtype: list
    """
    data = pd.read_excel(input_path)
    required = ['Time', 'Duration', 'Label']

    if not all(col in data.columns for col in required):
        raise ValueError("Excel file missing required columns")

    return data[required].values.tolist()


def save_events_to_excel(events, output_path):
    """
    Save events to Excel.

    :param events: Events data
    :type events: list
    :param output_path: Output file path
    :type output_path: str
    """
    df = pd.DataFrame(events, columns=['Time', 'Duration', 'Label'])
    df.to_excel(output_path, index=False)
    print(f"Events saved: {output_path}")