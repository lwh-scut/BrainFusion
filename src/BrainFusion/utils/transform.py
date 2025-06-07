import json
import re

import matplotlib.pyplot as plt
import mne.io
import numpy as np
from mne._freesurfer import get_mni_fiducials
from mne.io._digitization import _format_dig_points
from mne.io.constants import FIFF
from mne.transforms import apply_trans

from BrainFusion.io.File_IO import create_standard_bdf_file, read_bdf, read_snirf


def dict_to_snirf_raw(data_dict):
    """
    Convert SNIRF data dictionary to MNE Raw object.

    :param data_dict: SNIRF data structure dictionary
    :type data_dict: dict
    :return: MNE Raw object
    :rtype: mne.io.Raw
    """
    # Extract data components
    ch_names = list(data_dict['ch_names'])
    data = data_dict['data']
    times = data_dict['time']
    sfreq = 1.0 / np.mean(np.diff(times))

    source_pos_3d = data_dict['loc']['sourcePos3D']
    detector_pos_3d = data_dict['loc']['detectorPos3D']

    # Create MNE info structure
    info = mne.create_info(ch_names, sfreq, ch_types='fnirs_cw_amplitude')
    raw = mne.io.RawArray(data.T, info)

    # Configure channel locations
    for i, ch_name in enumerate(ch_names):
        match = re.match(r"S(\d+)_D(\d+)\s+(\d+)", ch_name)
        if match:
            source_idx = int(match.group(1))
            detector_idx = int(match.group(2))
            wavelength = int(match.group(3))
        loc = np.zeros(12)
        loc[3:6] = source_pos_3d[source_idx - 1] / 100  # Convert cm to meters
        loc[6:9] = detector_pos_3d[detector_idx - 1] / 100
        loc[0:3] = (loc[3:6] + loc[6:9]) / 2
        loc[9] = wavelength
        raw.info['chs'][i]['loc'] = loc
        raw.info['chs'][i]['coord_frame'] = FIFF.FIFFV_COORD_UNKNOWN

    # Add annotations
    events = data_dict.get('events', [])
    for event in events:
        onset, duration, label = event
        raw.annotations.append(onset, duration, label)

    return raw


def raw_to_dict(raw):
    """
    Convert MNE Raw object to SNIRF-compatible dictionary.

    :param raw: MNE Raw object
    :type raw: mne.io.Raw
    :return: SNIRF data structure dictionary
    :rtype: dict
    """
    # Extract signal data
    ch_names = raw.info['ch_names']
    sfreq = raw.info['sfreq']
    data, times = raw.get_data(return_times=True)

    # Extract source and detector positions
    source_pos_3d = []
    detector_pos_3d = []
    sd_pairs = []
    wavelengths = []

    for ch_idx, ch in enumerate(raw.info['chs']):
        loc = ch['loc']
        source_pos = loc[3:6]
        detector_pos = loc[6:9]
        wavelength = loc[9]

        # Store unique positions
        if source_pos.tolist() not in source_pos_3d:
            source_pos_3d.append(source_pos.tolist())
        if detector_pos.tolist() not in detector_pos_3d:
            detector_pos_3d.append(detector_pos.tolist())
        if wavelength not in wavelengths:
            wavelengths.append(wavelength)

        # Record pair indices
        source_idx = source_pos_3d.index(source_pos.tolist()) + 1
        detector_idx = detector_pos_3d.index(detector_pos.tolist()) + 1
        sd_pairs.append((source_idx, detector_idx))

    # Extract event annotations
    events = []
    if raw.annotations is not None:
        for annotation in raw.annotations:
            onset = annotation['onset']
            duration = annotation['duration']
            label = annotation['description']
            events.append([onset, duration, label])

    # Extract landmarks
    landmark_pos_3d = []
    landmark_labels = []
    if 'dig' in raw.info and raw.info['dig'] is not None:
        for dig_point in raw.info['dig']:
            if dig_point['kind'] == FIFF.FIFFV_POINT_EEG:
                landmark_pos_3d.append(dig_point['r'])
                landmark_labels.append(f"Point {len(landmark_labels) + 1}")
            elif dig_point['kind'] in (FIFF.FIFFV_POINT_LPA,
                                       FIFF.FIFFV_POINT_NASION,
                                       FIFF.FIFFV_POINT_RPA):
                landmark_pos_3d.append(dig_point['r'])
                landmark_labels.append({
                                           FIFF.FIFFV_POINT_LPA: "LPA",
                                           FIFF.FIFFV_POINT_NASION: "Nasion",
                                           FIFF.FIFFV_POINT_RPA: "RPA",
                                       }[dig_point['kind']])

    # Build dictionary
    return {
        'data': data,
        'time': times,
        'events': events,
        'ch_names': ch_names,
        'srate': sfreq,
        'loc': {
            'sourcePos3D': np.array(source_pos_3d) * 100,
            'detectorPos3D': np.array(detector_pos_3d) * 100,
            'landmarkPos3D': np.array(landmark_pos_3d) * 100,
        },
        'sd': sd_pairs,
        'wavelengths': wavelengths,
    }


def raw_to_dict_eeg(raw):
    """
    Convert EEG Raw object to simplified dictionary.

    :param raw: MNE Raw object
    :type raw: mne.io.Raw
    :return: EEG data structure dictionary
    :rtype: dict
    """
    # Extract signal data
    ch_names = raw.info['ch_names']
    sfreq = raw.info['sfreq']
    data, times = raw.get_data(return_times=True)

    # Extract event annotations
    events = []
    if raw.annotations is not None:
        for annotation in raw.annotations:
            onset = annotation['onset']
            duration = annotation['duration']
            label = annotation['description']
            events.append([onset, duration, label])

    # Build dictionary
    return {
        'data': data,
        'time': times,
        'events': events,
        'ch_names': ch_names,
        'srate': sfreq,
    }


def dict_to_info(data_dict, filePath):
    """
    Save metadata to JSON file excluding large data arrays.

    :param data_dict: Data structure dictionary
    :type data_dict: dict
    :param filePath: JSON output file path
    :type filePath: str
    """
    # Create metadata dictionary
    info_dict = {key: value for key, value in data_dict.items()
                 if key not in ['data', 'time', 'events']}

    # Serialization helper
    def serialize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        raise TypeError(f"Unserializable type: {type(obj)}")

    # Save to JSON
    with open(filePath, 'w', encoding='utf-8') as f:
        json.dump(info_dict, f, default=serialize, ensure_ascii=False, indent=4)


def read_info(filePath):
    """
    Read JSON metadata file into dictionary.

    :param filePath: JSON input file path
    :type filePath: str
    :return: Metadata dictionary
    :rtype: dict
    """
    with open(filePath, 'r', encoding='utf-8') as f:
        return json.load(f)

