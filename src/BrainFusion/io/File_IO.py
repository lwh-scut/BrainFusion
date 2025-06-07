# -*- coding: utf-8 -*-
"""
File Input/Output Module

Provides functions for reading, writing and converting various biomedical file formats
including EEG, ECG, fNIRS and more.
"""

import json
import os
import sys
from datetime import datetime
from tkinter import Tk, filedialog
import pyedflib
import mne
import numpy as np
import pandas as pd
import snirf
from PyQt5.QtWidgets import QFileDialog
from scipy.io import loadmat, savemat
import csv
import ast
import re

from BrainFusion.io.neuracle_lib.readbdfdata import readbdfdata
from BrainFusion.io.snirf_io import create_snirf_file


def create_data_dict(data, srate, nchan=None, ch_names=None, events=None,
                     type='eeg', montage=None, is_save=False, save_path=None,
                     save_filestyle='mat'):
    """
    Creates a standardized data dictionary for biomedical signals

    :param data: Signal data array (channels x time points)
    :type data: np.ndarray or list
    :param srate: Sampling rate in Hz
    :type srate: float
    :param nchan: Number of channels, defaults to None (inferred from data)
    :type nchan: int, optional
    :param ch_names: Channel names, defaults to None
    :type ch_names: list[str], optional
    :param events: Event markers in format [[time, duration, label], ...], defaults to None
    :type events: list[list], optional
    :param type: Data type ('eeg', 'ecg', 'fnirs', etc.), defaults to 'eeg'
    :type type: str, optional
    :param montage: Sensor montage information, defaults to None
    :type montage: str, optional
    :param is_save: Whether to save the data dictionary, defaults to False
    :type is_save: bool, optional
    :param save_path: Path to save file, defaults to None
    :type save_path: str, optional
    :param save_filestyle: File format for saving ('mat', 'csv', 'json', etc.), defaults to 'mat'
    :type save_filestyle: str, optional
    :return: Standardized data dictionary
    :rtype: dict
    """
    data_dict = {}
    data_dict['data'] = data
    data_dict['srate'] = srate
    data_dict['events'] = events
    data_dict['nchan'] = nchan
    data_dict['ch_names'] = ch_names
    data_dict['type'] = type
    data_dict['montage'] = montage
    save_prepare(data_dict)
    if is_save:
        save_file(data=data_dict, save_path=save_path, save_filestyle=save_filestyle)
    return data_dict


def reduce_dimensions(arr):
    """
    Reduces array dimensions by removing single-dimensional entries

    :param arr: Input array
    :type arr: np.ndarray
    :return: Reduced-dimension array
    :rtype: np.ndarray or scalar
    """
    num_dims = arr.ndim
    num_elements = np.prod(arr.shape)
    if num_elements <= num_dims:
        arr = np.squeeze(arr)
    if arr.ndim == 0:
        arr = arr.item()
    return arr


def remove_keys(d, char):
    """
    Removes dictionary keys containing a specific character

    :param d: Input dictionary
    :type d: dict
    :param char: Character to exclude from keys
    :type char: str
    :return: Filtered dictionary
    :rtype: dict
    """
    return {k: v for k, v in d.items() if char not in k}


def check_data_dict(data_dict):
    """
    Validates the structure of a data dictionary

    :param data_dict: Data dictionary to validate
    :type data_dict: dict
    :return: True if valid
    :rtype: bool
    :raises TypeError: If input is not a dictionary
    :raises ValueError: If required fields are missing or invalid
    """
    if isinstance(data_dict, dict):
        if isinstance(data_dict['data'], np.ndarray) or isinstance(data_dict['data'], list):
            if data_dict['srate']:
                return True
            else:
                raise ValueError("srate should not be None")
        else:
            raise ValueError("data should not be None")
    else:
        raise TypeError("data_dict should be a dict")


def find_bids_files(root_folder):
    """
    Scans a root folder for BIDS-formatted data files

    :param root_folder: Root directory to scan
    :type root_folder: str
    :return: List of found data files with associated label files
    :rtype: list[dict]
    """
    data_and_label_files = []
    for sub_folder in os.listdir(root_folder):
        sub_folder_path = os.path.join(root_folder, sub_folder)
        if os.path.isdir(sub_folder_path):
            data_and_labels = []
            if re.search(r'sub-\d+', sub_folder):
                for data_folder in os.listdir(sub_folder_path):
                    data_folder_path = os.path.join(sub_folder_path, data_folder)
                    if os.path.isdir(data_folder_path):
                        data_file_path = None
                        label_file_path = None
                        data_type = None
                        for data_file_name in os.listdir(data_folder_path):
                            if re.search(r'eeg|ecg|emg|nirs|other', data_file_name):
                                data_file_path = os.path.join(data_folder_path, data_file_name)
                                data_type = re.search(r'eeg|ecg|emg|nirs|other', data_file_name).group() + "_data_file"
                            if 'event' in data_file_name:
                                label_file_path = os.path.join(data_folder_path, data_file_name)

                        if data_file_path:
                            data_and_labels.append({data_type: data_file_path, 'label_file': label_file_path})

                    if data_and_labels:
                        data_and_label_files.append(data_and_labels)

    return data_and_label_files


def read_one_bids_file(path_list):
    """
    Reads data from a list of BIDS file paths

    :param path_list: List of file paths in BIDS format
    :type path_list: list[dict]
    :return: Dictionary of loaded data
    :rtype: dict
    :raises ValueError: If path_list is empty
    """
    data_dict = {}
    if path_list:
        for filePath in path_list:
            if filePath['label_file'] is None:
                for key in filePath.keys():
                    if 'data' in key:
                        new_key = key.split('_')[0]
                        data_dict[new_key] = read_file_by_path(filePath[key])
            else:
                for key in filePath.keys():
                    if 'data' in key:
                        new_key = key.split('_')[0]
                        data_dict[new_key] = read_file_by_path((filePath[key], filePath['label_file']))
    else:
        raise ValueError("path_list is empty.")
    return data_dict


def read_file_by_qt(widget, path=None):
    """
    Reads a file using a Qt file dialog

    :param widget: Parent Qt widget
    :type widget: QWidget
    :param path: Predefined file path, defaults to None (opens dialog)
    :type path: str, optional
    :return: Loaded data and file path
    :rtype: tuple(object, str)
    """
    data = None
    if path is None:
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        try:
            path, _ = QFileDialog.getOpenFileNames(widget, 'Open Files', '',
                                                   'All Files (*);;Two Bdf Files (*.bdf);;Edf '
                                                   'File (*.edf);;Text File (*.txt);;Json '
                                                   'File(*.json);;Mat File(*.mat);;NIRS File(*.nirs)',
                                                   options=options)
        except Exception as e:
            print(e)
    if len(path) == 1:
        file_type = path[0].split('.')[-1]
        if file_type == 'edf':
            data = read_edf(path[0])
        elif file_type == 'bdf':
            data = read_bdf(path[0])
        elif file_type == 'nirs':
            data = read_nirs(path[0])
        elif file_type == 'csv':
            data = read_csv(path[0])
        elif file_type == 'txt':
            data = read_txt(path[0])
        elif file_type == 'json':
            data = read_json(path[0])
        elif file_type == 'mat':
            data = read_mat(path[0])
        elif file_type == 'ecg':
            data = read_ecg(path[0])
        elif file_type == 'xlsx':
            data = read_xlsx(path[0])
        elif file_type == 'snirf':
            data = read_snirf(path[0])
    elif len(path) == 2:
        if path[0].split('.')[-1] == 'bdf':
            data = read_neuracle_bdf(path, is_data_transform=True)
    return data, path


def read_file_by_mne(widget, path=None):
    """
    Reads a file using MNE-Python via Qt dialog

    :param widget: Parent Qt widget
    :type widget: QWidget
    :param path: Predefined file path, defaults to None (opens dialog)
    :type path: str, optional
    :return: MNE Raw object and file path
    :rtype: tuple(mne.io.Raw, str)
    """
    data = None
    if path is None:
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        try:
            path, _ = QFileDialog.getOpenFileNames(widget, 'Open Files', '',
                                                   'All Files (*);;Two Bdf Files (*.bdf);;Edf '
                                                   'File (*.edf);;Text File (*.txt);;Json '
                                                   'File(*.json);;Mat File(*.mat);;NIRS File(*.nirs)',
                                                   options=options)
        except Exception as e:
            print(e)
    if len(path) == 1:
        file_type = path[0].split('.')[-1]
        if file_type == 'snirf':
            data = mne.io.read_raw_snirf(path[0])
    return data, path


def read_file_by_path(path=None):
    """
    Reads a file from a given path

    :param path: File path or tuple of paths, defaults to None
    :type path: str or tuple, optional
    :return: Loaded data
    :rtype: dict or object
    """
    data = None
    if len(path) == 2:
        if path[0].split('.')[-1] == 'bdf':
            data = read_neuracle_bdf(path, is_data_transform=True)
    else:
        file_type = path.split('.')[-1]
        if file_type == 'edf':
            data = read_edf(path)
        elif file_type == 'bdf':
            data = read_bdf(path)
        elif file_type == 'nirs':
            data = read_nirs(path)
        elif file_type == 'csv':
            data = read_csv(path)
        elif file_type == 'txt':
            data = read_txt(path)
        elif file_type == 'json':
            data = read_json(path)
        elif file_type == 'mat':
            data = read_mat(path)
        elif file_type == 'ecg':
            data = read_ecg(path)
        elif file_type == 'snirf':
            data = read_snirf(path)

    return data


def read_file(path=None):
    """
    Reads a file using Tkinter file dialog

    :param path: Predefined file path, defaults to None (opens dialog)
    :type path: str, optional
    :return: Loaded data
    :rtype: dict
    """
    if path is None:
        root = Tk()
        root.withdraw()
        try:
            path = filedialog.askopenfilenames(initialdir='/', title='Select one csv files',
                                               filetypes=(
                                                   ("all files", "*"), ("one csv file", "*.csv"),
                                                   ("one txt file", "*.txt"),
                                                   ("one json file", "*.json"), ("one mat file", "*.mat"),
                                                   ("one edf file", "*.edf"), ("one bdf file", "*.bdf"),
                                                   ("one nirs file", "*.nirs")))
        except Exception as e:
            print(e)
    data = None
    if len(path) == 1:
        file_type = path[0].split('.')[-1]
        print(file_type)
        if file_type == 'edf':
            data = read_edf(path[0])
        elif file_type == 'bdf':
            data = read_bdf(path[0])
        elif file_type == 'nirs':
            data = read_nirs(path[0])
        elif file_type == 'csv':
            data = read_csv(path[0])
        elif file_type == 'txt':
            data = read_txt(path[0])
        elif file_type == 'json':
            data = read_json(path[0])
        elif file_type == 'mat':
            data = read_mat(path[0])
        elif file_type == 'ecg':
            data = read_ecg(path[0])
    elif len(path) == 2:
        if path[0].split('.')[-1] == 'bdf':
            data = read_neuracle_bdf(path, is_data_transform=True)
            print(data['events'])
    return data


def read_neuracle_bdf(path=None, is_data_transform=True):
    """
    Reads Neuracle BDF data files

    :param path: Path to BDF file(s), defaults to None (opens dialog)
    :type path: str or tuple, optional
    :param is_data_transform: Whether to transform data to standard format, defaults to True
    :type is_data_transform: bool, optional
    :return: Loaded BDF data
    :rtype: dict
    :raises TypeError: For invalid file formats or missing files
    """

    def check_files_format(path):
        filename = []
        pathname = []
        if len(path) == 0:
            raise TypeError('please select valid file')

        elif len(path) == 1:
            (temppathname, tempfilename) = os.path.split(path[0])
            if 'edf' in tempfilename:
                filename.append(tempfilename)
                pathname.append(temppathname)
                return filename, pathname
            elif 'bdf' in tempfilename:
                raise TypeError('unsupport only one neuracle-bdf file')
            else:
                raise TypeError('not support such file format')

        else:
            temp = []
            temppathname = r''
            evtfile = []
            idx = np.zeros((len(path) - 1,))
            for i, ele in enumerate(path):
                (temppathname, tempfilename) = os.path.split(ele)
                if 'data' in tempfilename or 'eeg' in tempfilename or 'emg' in tempfilename:
                    temp.append(tempfilename)
                    if len(tempfilename.split('.')) > 2:
                        try:
                            idx[i] = (int(tempfilename.split('.')[1]))
                        except:
                            raise TypeError('no such kind file')
                    else:
                        idx[i] = 0
                elif 'evt' in tempfilename or 'event' in tempfilename:
                    evtfile.append(tempfilename)

            pathname.append(temppathname)
            datafile = [temp[i] for i in np.argsort(idx)]

            if len(evtfile) == 0:
                raise TypeError('not found evt.bdf file')

            if len(datafile) == 0:
                raise TypeError('not found data.bdf file')
            elif len(datafile) > 1:
                print('current readbdfdata() only support continue one data.bdf ')
                return filename, pathname
            else:
                filename.append(datafile[0])
                filename.append(evtfile[0])
                return filename, pathname

    def data_transform(data):
        result = create_data_dict(data=data['data'],
                                  srate=data['srate'],
                                  ch_names=data['ch_names'],
                                  nchan=data['nchan'],
                                  events=data['events'])
        return result

    if path is None:
        root = Tk()
        root.withdraw()
        try:
            path = filedialog.askopenfilenames(initialdir='/', title='Select two bdf files',
                                               filetypes=(("two bdf files", "*.bdf"), ("one edf files", "*.edf")))
        except Exception as e:
            print(e)
    data = None
    if path is not None:
        filename, pathname = check_files_format(path)
        data = readbdfdata(filename, pathname)
        if is_data_transform:
            data = data_transform(data)
    return data


def read_csv(path=None):
    """
    Reads data from a CSV file

    :param path: Path to CSV file, defaults to None (opens dialog)
    :type path: str, optional
    :return: Loaded data dictionary
    :rtype: dict
    """
    if path is None:
        root = Tk()
        root.withdraw()
        try:
            path = filedialog.askopenfilenames(initialdir='/', title='Select one csv files',
                                               filetypes=(("one csv file", "*.csv"),))[0]
        except Exception as e:
            print(e)
    data = None
    data_list = []
    if path is not None:
        with open(path, 'r') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                row['data'] = row['data'].replace('\n  ', ',').replace('\n ', ',').replace(' ', ',')
                row['nchan'] = int(row['nchan']) if 'nchan' in row else row['data'].shape[0]
                row['srate'] = int(float(row['srate']))
                row['ch_names'] = ast.literal_eval(row['ch_names']) if 'ch_names' in row else []
                row['events'] = row['events'].replace('      ', ',').replace('\n  ', ',').replace('\n ', ',').replace(
                    ' ', ',')
                row['events'] = ast.literal_eval(row['events']) if 'events' in row else []
                data_list.append(row)
        data = data_list[0]
    return data


def read_txt(path=None):
    """
    Reads data from a text file

    :param path: Path to text file, defaults to None (opens dialog)
    :type path: str, optional
    :return: Loaded data
    :rtype: dict
    """
    if path is None:
        root = Tk()
        root.withdraw()
        try:
            path = filedialog.askopenfilenames(initialdir='/', title='Select one txt files',
                                               filetypes=(("one txt file", "*.txt"),))[0]
        except Exception as e:
            print(e)
    data = None
    if path is not None:
        with open(path, 'r') as json_file:
            data = json.load(json_file)
    return data


def read_npy(path=None):
    """
    Reads data from a NumPy .npy file

    :param path: Path to .npy file, defaults to None (opens dialog)
    :type path: str, optional
    :return: Loaded array
    :rtype: np.ndarray
    """
    if path is None:
        root = Tk()
        root.withdraw()
        try:
            path = filedialog.askopenfilenames(initialdir='/', title='Select one npy files',
                                               filetypes=(("one npy file", "*.npy"),))[0]
        except Exception as e:
            print(e)
    data = None
    if path is not None:
        print(path)
        data = np.load(path)
    return data


def read_mat(path=None):
    """
    Reads data from a MATLAB .mat file

    :param path: Path to .mat file, defaults to None (opens dialog)
    :type path: str, optional
    :return: Loaded data dictionary
    :rtype: dict
    """
    if path is None:
        root = Tk()
        root.withdraw()
        try:
            path = filedialog.askopenfilenames(initialdir='/', title='Select one mat files',
                                               filetypes=(("one mat file", "*.mat"),))[0]
        except Exception as e:
            print(e)
    data = None
    if path is not None:
        print(path)
        data = loadmat(path)
        data = remove_keys(data, '__')
        print("Variables in MATLAB files:", data.keys())
        for key in data.keys():
            data[key] = reduce_dimensions(data[key])
            data[key] = convert_to_list(data[key])
            if data[key] == '' or data[key] == []:
                data[key] = None
            if key == 'ch_names':
                data[key] = [name.replace(' ', '') for name in data[key]]
            try:
                if key == 'events' and data[key] is not None:
                    for i, event, duration, label in enumerate(data[key]):
                        if isinstance(event, str):
                            event = float(event.replace(' ', ''))
                        if isinstance(duration, str):
                            duration = float(duration.replace(' ', ''))
                        if isinstance(label, str):
                            label = label.replace(' ', '')
                        data[key][i] = [event, duration, label]
            except:
                pass
    return data


def read_edf(path=None):
    """
    Reads data from an EDF file

    :param path: Path to EDF file, defaults to None (opens dialog)
    :type path: str, optional
    :return: Loaded data dictionary
    :rtype: dict
    """
    if path is None:
        root = Tk()
        root.withdraw()
        try:
            path = filedialog.askopenfilenames(initialdir='/', title='Select one edf files',
                                               filetypes=(("one edf file", "*.edf"),))[0]
        except Exception as e:
            print(e)
    data = None
    if path is not None:
        edf_file = pyedflib.EdfReader(path)
        edf_data = []
        nchan = edf_file.signals_in_file
        for chan in range(nchan):
            edf_data.append(edf_file.readSignal(chan, digital=True) * 0.000001)
        data_dict = {}
        data_dict['极好data'] = edf_data
        data_dict['srate'] = edf_file.getSampleFrequencies()[0]
        data_dict['events'] = edf_file.readAnnotations()
        data_dict['nchan'] = edf_file.signals_in_file
        data_dict['ch_names'] = [chan.replace('.', '') for chan in edf_file.getSignalLabels()]
        data_dict['units'] = [edf_file.getPhysicalDimension(i) for i in range(nchan)]
        data_dict['type'] = 'eeg'
        data_dict['montage'] = None
        data = data_dict
        edf_file.close()
    return data


def read_bdf(path=None, type='eeg', montage=None):
    """
    Reads data from a BDF file

    :param path: Path to BDF file, defaults to None (opens dialog)
    :type path: str, optional
    :param type: Data type ('eeg', 'ecg', etc.), defaults to 'eeg'
    :type type: str, optional
    :param montage: Sensor montage information, defaults to None
    :type montage: str, optional
    :return: Loaded data dictionary
    :rtype: dict
    """
    if path is None:
        root = Tk()
        root.withdraw()
        try:
            path = filedialog.askopenfilenames(initialdir='/', title='Select one bdf files',
                                               filetypes=(("one bdf file", "*.bdf"),))[0]
        except Exception as e:
            print(e)
    data = None
    if path is not None:
        bdf_file = pyedflib.EdfReader(path)
        bdf_data = []
        nchan = bdf_file.signals_in_file
        for chan in range(nchan):
            bdf_data.append(bdf_file.readSignal(chan, digital=False) * 0.000001)
        data_dict = {}
        data_dict['data'] = bdf_data
        data_dict['srate'] = bdf_file.getSampleFrequencies()[0]
        data_dict['events'] = np.array(bdf_file.readAnnotations()).T.tolist()
        for idx, event in enumerate(data_dict['events']):
            time, duration, label = event
            label = str(int(float(label)))
            time = float(time)
            duration = float(duration)
            data_dict['events'][idx] = [time, duration, label]
        data_dict['nchan'] = bdf_file.signals_in_file
        data_dict['ch_names'] = [chan.replace('.', '') for chan in bdf_file.getSignalLabels()]
        data_dict['units'] = [bdf_file.getPhysicalDimension(i) for i in range(nchan)]
        data_dict['type'] = type
        data_dict['montage'] = montage
        data_dict['file_info'] = {'PhysicalMaximum': bdf_file.getPhysicalMaximum(),
                                  'PhysicalMinimum': bdf_file.getPhysicalMinimum(),
                                  'DigitalMaximum': bdf_file.getDigitalMaximum(),
                                  'DigitalMinimum': bdf_file.getDigitalMinimum()}
        bdf_file.close()
        data = data_dict.copy()
    return data


def read_minilab_snirf(path=None, type='fnirs'):
    """
    Reads SNIRF files with MiniLab-specific extensions

    :param path: Path to SNIRF file, defaults to None
    :type path: str, optional
    :param type: Data type, defaults to 'fnirs'
    :type type: str, optional
    :return: Loaded data dictionary with additional metadata
    :rtype: dict
    """
    snirf_file = snirf.loadSnirf(path)
    data_dict = {}
    data_dict['data'] = snirf_file.nirs[0].data[0].dataTimeSeries
    data_dict['time'] = snirf_file.nirs[0].data[0].time
    data_dict['srate'] = 1.0 / np.mean(np.diff(data_dict['time']))
    if len(snirf_file.nirs[0].stim) > 0:
        all_events = []
        for stim in snirf_file.nirs[0].stim:
            if stim.data is not None and len(stim.data) > 0:
                for event in stim.data:
                    time, duration, label = event
                    label = str(int(float(label)))
                    time = float(time)
                    duration = float(duration)
                    all_events.append([time, duration, label])

        all_events.sort(key=lambda x: x[0])
        data_dict['events'] = all_events if all_events else None
    else:
        data_dict['events'] = None

    data_dict['nchan'] = snirf_file.nirs[0].data[0].dataTimeSeries.shape[1]
    data_dict['ch_names'] = snirf_file.nirs[0].probe.landmarkLabels
    data_dict['type'] = type
    loc = {
        'sourcePos3D': snirf_file.nirs[0].probe.sourcePos3D,
        'detectorPos3D': snirf_file.nirs[0].probe.detectorPos3D,
        'landmarkPos3D': snirf_file.nirs[0].probe.landmarkPos3D
    }
    data_dict['loc'] = loc

    sd_pairs = []
    for ch_name in data_dict['ch_names'][:data_dict['nchan'] // 2]:
        sd_part = ch_name.split(" ")[0]
        source, detector = sd_part.split("_")
        source_idx = int(source[1:])
        detector_idx = int(detector[1:])
        sd_pairs.append((source_idx, detector_idx))
    data_dict['sd'] = sd_pairs
    data_dict['wavelengths'] = snirf_file.nirs[0].probe.wavelengths

    return data_dict


def read_minilab_bdf(path, type='eeg', montage=''):
    """
    Reads BDF files with MiniLab-specific extensions

    :param path: Path to BDF file
    :type path: str
    :param type: Data type, defaults to 'eeg'
    :type type: str, optional
    :param montage: Sensor montage information, defaults to ''
    :type montage: str, optional
    :return: Loaded data dictionary with additional metadata
    :rtype: dict
    """
    bdf_file = pyedflib.EdfReader(path)
    bdf_data = []
    nchan = bdf_file.signals_in_file
    for chan in range(nchan):
        bdf_data.append(bdf_file.readSignal(chan, digital=False) * 0.000001)
    data_dict = {}
    data_dict['data'] = bdf_data
    data_dict['srate'] = bdf_file.getSampleFrequencies()[0]
    data_dict['events'] = np.array(bdf_file.readAnnotations()).T.tolist()
    for idx, event in enumerate(data_dict['events']):
        time, duration, label = event
        label = str(int(float(label)))
        time = float(time)
        duration = float(duration)
        data_dict['events'][idx] = [time, duration, label]
    data_dict['nchan'] = bdf_file.signals_in_file
    data_dict['ch_names'] = [chan.replace('.', '') for chan in bdf_file.getSignalLabels()]
    data_dict['units'] = [bdf_file.getPhysicalDimension(i) for i in range(nchan)]
    data_dict['type'] = type
    data_dict['montage'] = montage
    data_dict['file_info'] = {'PhysicalMaximum': bdf_file.getPhysicalMaximum(),
                              'PhysicalMinimum': bdf_file.getPhysicalMinimum(),
                              'DigitalMaximum': bdf_file.getDigitalMaximum(),
                              'DigitalMinimum': bdf_file.getDigitalMinimum()}
    bdf_file.close()
    return data_dict


def read_json(path=None):
    """
    Reads data from a JSON file

    :param path: Path to JSON file, defaults to None (opens dialog)
    :type path: str, optional
    :return: Loaded data
    :rtype: dict
    """
    if path is None:
        root = Tk()
        root.withdraw()
        try:
            path = filedialog.askopenfilenames(initialdir='/', title='Select one json files',
                                               filetypes=(("one json file", "*.json"),))[0]
        except Exception as e:
            print(e)
    data = None
    if path is not None:
        with open(path, 'r') as json_file:
            data = json.load(json_file)
    return data


def read_snirf(path=None, type='fnirs'):
    """
    Reads data from a SNIRF file

    :param path: Path to SNIRF file, defaults to None (opens dialog)
    :type path: str, optional
    :param type: Data type, defaults to 'fnirs'
    :type type: str, optional
    :return: Loaded data dictionary
    :rtype: dict
    """
    if path is None:
        root = Tk()
        root.withdraw()
        try:
            path = filedialog.askopenfilenames(initialdir='/', title='Select one snirf files',
                                               filetypes=(("one snirf file", "*.snirf"),))[0]
        except Exception as e:
            print(e)
    data = None
    if path is not None:
        data = read_minilab_snirf(path, type=type)
    return data


def read_nirs(path=None):
    """
    Reads data from a .nirs file (NIRx format)

    :param path: Path to .nirs file, defaults to None (opens dialog)
    :type path: str, optional
    :return: Loaded data dictionary
    :rtype: dict
    """

    def trans_list_to_str(channel_list):
        str_list = []
        nchan = len(channel_list)
        for i in range(0, int(nchan / 2)):
            ch_name = str(channel_list[i][0]) + '_' + str(channel_list[i][1]) + '_' + 'hbo'
            str_list.append(ch_name)
        for i in range(int(nchan / 2), nchan):
            ch_name = str(channel_list[i][0]) + '_' + str(channel_list[i][1]) + '_' + 'hbr'
            str_list.append(ch_name)
        return str_list

    if path is None:
        root = Tk()
        root.withdraw()
        try:
            path = filedialog.askopenfilenames(initialdir='/', title='Select one mat files',
                                               filetypes=(("one nirs file", "*.nirs"),))[0]
        except Exception as e:
            print(e)
    data = None
    if path is not None:
        mat_data = loadmat(path)
        freqs = mat_data['SD']['Lambda'][0][0][0]
        if isinstance(freqs, np.ndarray):
            freqs = freqs.tolist()
        events = mat_data['s']
        events = [(i, j, events[i][j]) for i in range(len(events)) for j in range(len(events[i])) if events[i][j] != 0]

        ch_names = trans_list_to_str(mat_data['ml'])
        nchan = len(mat_data['ml'])
        src_pos = mat_data['SD']['SrcPos'][0][0]
        det_pos = mat_data['SD']['DetPos'][0][0]
        data = {}
        data['data'] = mat_data['d'].T
        data['freqs'] = freqs
        data['srate'] = 11
        data['nchan'] = nchan
        data['ch_names'] = ch_names
        data['events'] = events
        data['type'] = 'fnirs'
        data['montage'] = None
        data['loc'] = [src_pos, det_pos]
    return data


from ishneholterlib import Holter


def read_ecg(file_path):
    """
    Reads data from an ECG (ISHNE Holter) file

    :param file_path: Path to ECG file
    :type file_path: str
    :return: ECG data dictionary
    :rtype: dict
    """
    ecg = Holter(file_path)
    ecg.load_data()
    nchan = ecg.nleads
    ch_names = [ecg.lead[i].spec_str() for i in range(nchan)]
    data = []
    for chan in range(nchan):
        data.append(ecg.lead[chan].data)
    srate = ecg.sr

    ecg_data = {}
    ecg_data['data'] = data
    ecg_data['events'] = None
    ecg_data['srate'] = srate
    ecg_data['nchan'] = nchan
    ecg_data['ch_names'] = ch_names
    ecg_data['type'] = 'ecg'
    ecg_data['montage'] = ''
    return ecg_data


def read_xlsx(file_path):
    """
    Reads features from an Excel (.xlsx) file

    :param file_path: Path to Excel file
    :type file_path: str
    :return: Feature dictionary
    :rtype: dict
    """
    df = pd.read_excel(file_path)
    feature_dict = {
        "ch_names": df['Channel'].tolist(),
        "feature": {},
        "type": df['Type'].unique()[0]
    }

    feature_columns = [col for col in df.columns if col not in ['Channel', 'Type']]

    for feat in feature_columns:
        feature_dict['feature'][feat] = []

    for idx, row in df.iterrows():
        for feat in feature_columns:
            feature_dict['feature'][feat].append(row[feat])

    return feature_dict


def array_to_string(array):
    """
    Converts a NumPy array to string representation

    :param array: Input array
    :type array: np.ndarray
    :return: String representation of array
    :rtype: str
    """
    s = []
    for i in array:
        s.append(np.array2string(i, max_line_width=np.inf, separator=','))
    return s


def save_prepare(data):
    """
    Prepares data for saving by converting to lists and handling empty values

    :param data: Data to prepare
    :type data: dict
    :return: Prepared data
    :rtype: dict
    """
    for key in data.keys():
        if isinstance(data[key], np.ndarray):
            data[key] = convert_to_list(data[key])
        elif isinstance(data[key], frozenset):
            data[key] = convert_to_list(data[key])
    data = {key: '' if value is None else value for key, value in data.items()}
    return data


def save_feature(data, save_path=None, save_filestyle='csv'):
    """
    Saves feature data to a file in specified format

    :param data: Feature data (pandas DataFrame)
    :type data: pd.DataFrame
    :param save_path: Path to save file, defaults to None
    :type save_path: str, optional
    :param save_filestyle: File format ('csv', 'xlsx', 'tsv', 'mat'), defaults to 'csv'
    :type save_filestyle: str, optional
    """
    if save_path:
        if save_filestyle == 'csv':
            data.to_csv(save_path, index=False)
        elif save_filestyle == 'xlsx':
            data.to_excel(save_path, index=False)
        elif save_filestyle == 'tsv':
            data.to_csv(save_path, sep='\t', index=False)
        elif save_filestyle == 'mat':
            savemat(save_path, {'data': data.values})
        else:
            print("Unsupported file format.")
    else:
        print("Please provide a save path.")


def save_file(data, save_path=None, save_filestyle='mat'):
    """
    Saves data to a file in specified format

    :param data: Data to save
    :type data: dict
    :param save_path: Path to save file, defaults to None (opens dialog)
    :type save_path: str, optional
    :param save_filestyle: File format ('mat', 'csv', 'json', 'txt'), defaults to 'mat'
    :type save_filestyle: str, optional
    """
    if save_path:
        if save_filestyle == 'csv':
            save_csv(data, save_path)
        elif save_filestyle == 'txt':
            save_txt(data, save_path)
        elif save_filestyle == 'mat':
            save_mat(data, save_path)
        elif save_filestyle == 'json':
            save_json(data, save_path)
    else:
        if save_filestyle == 'csv':
            save_csv(data)
        elif save_filestyle == 'txt':
            save_txt(data)
        elif save_filestyle == 'mat':
            save_mat(data)
        elif save_filestyle == 'json':
            save_json(data)


def convert_to_list(data):
    """
    Converts numpy arrays to Python lists recursively

    :param data: Input data (array or nested structure)
    :type data: any
    :return: Converted data
    :rtype: list or primitive
    """
    if isinstance(data, float):
        return data
    if isinstance(data, (np.ndarray, np.generic)):
        if data.ndim == 0:
            return data
        if data.dtype.kind in ('S', 'U'):
            return [x.decode('utf-8') if isinstance(x, bytes) else x for x in data.flat]
        return [convert_to_list(item) for item in data]
    if isinstance(data, (list, tuple)):
        return [convert_to_list(item) for item in data]
    return data


def dict_to_snirf(data_dict, output_path):
    """
    Converts a data dictionary to SNIRF file format

    :param data_dict: Data dictionary with required fields
    :type data_dict: dict
    :param output_path: Output file path
    :type output_path: str
    """
    # Prepare data
    data_time_series = np.array(data_dict.get('data'))
    time_points = np.array(data_dict.get('time'))
    loc = data_dict.get('loc', {})

    # Build measurement lists
    measurement_lists = []
    sd_pairs = data_dict.get('sd', [])
    wavelengths = data_dict.get('wavelengths', [])
    for i, (source_idx, detector_idx) in enumerate(sd_pairs):
        for wl_idx, wavelength in enumerate(wavelengths, start=1):
            measurement_lists.append({
                'sourceIndex': source_idx,
                'detectorIndex': detector_idx,
                'wavelengthIndex': wl_idx,
                'dataType': 1,
                'dataTypeIndex': len(measurement_lists) + 1,
            })

    # Prepare event lists
    stim_lists = []
    events = data_dict.get('events')
    if events is not None and len(events) > 0:
        for event in events:
            onset, duration, label = event
            stim_lists.append({
                'name': str(int(float(label))),
                'data': [[onset, duration, float(label)]],
                'dataLabels': ['onset', 'duration', 'label']
            })

    # Create SNIRF file
    create_snirf_file(
        filename=output_path,
        data_time_series=data_time_series,
        time_points=time_points,
        measurement_lists=measurement_lists,
        source_pos_3d=np.array(loc.get('sourcePos3D', [])),
        detector_pos_3d=np.array(loc.get('detectorPos3D', [])),
        wavelengths=np.array(wavelengths),
        landmark_pos_3d=np.array(loc.get('landmarkPos3D', [])),
        landmark_labels=data_dict.get('ch_names', []),
        stim_lists=stim_lists
    )
    print(f"SNIRF file saved to {output_path}")


def dict_to_bdf(data_dict, output_path):
    """
    Converts a data dictionary to BDF file format

    :param data_dict: Data dictionary with required fields
    :type data_dict: dict
    :param output_path: Output file path
    :type output_path: str
    """
    signals = np.array(data_dict['data']) * 1000000
    n_channels, n_samples = signals.shape
    sampling_frequency = data_dict['srate']
    channel_names = data_dict['ch_names']
    annotations = data_dict.get('events', None)

    file_info = data_dict.get('file_info', {})
    physical_mins = file_info.get('PhysicalMinimum', [-10000] * n_channels)
    physical_maxs = file_info.get('PhysicalMaximum', [10000] * n_channels)
    digital_mins = file_info.get('DigitalMinimum', [-32768] * n_channels)
    digital_maxs = file_info.get('DigitalMaximum', [32767] * n_channels)
    physical_dimensions = file_info.get('PhysicalDimensions', ['uV'] * n_channels)

    technician = data_dict.get('technician', '')
    equipment = data_dict.get('equipment', '')
    patient_code = data_dict.get('patient_code', '')
    patient_name = data_dict.get('patient_name', '')
    birthdate = data_dict.get('birthdate', None)
    admin_code = data_dict.get('admin_code', '')

    transducers = ['eeg'] * n_channels

    create_standard_bdf_file(
        file_name=output_path,
        n_channels=n_channels,
        signals=signals,
        channel_names=channel_names,
        physical_mins=physical_mins,
        physical_maxs=physical_maxs,
        digital_mins=digital_mins,
        digital_maxs=digital_maxs,
        sampling_frequency=sampling_frequency,
        technician=technician,
        equipment=equipment,
        patient_code=patient_code,
        patient_name=patient_name,
        birthdate=birthdate,
        admin_code=admin_code,
        annotations=annotations,
        transducers=transducers,
        physical_dimensions=physical_dimensions
    )
    print(f"BDF file saved to: {output_path}")


def save_csv(data, path=None):
    """
    Saves data to a CSV file

    :param data: Data to save
    :type data: dict
    :param path: Save path, defaults to None (opens dialog)
    :type path: str, optional
    :raises ValueError: If data validation fails
    """
    root = Tk()
    root.withdraw()
    if path is None:
        try:
            path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")],
                                                title="Select file save path")
        except Exception as e:
            print(e)
    if path:
        if check_data_dict(data):
            data = save_prepare(data)
            with open(path, 'w', newline='') as csvfile:
                csv_writer = csv.DictWriter(csvfile, fieldnames=data.keys())
                csv_writer.writeheader()
                csv_writer.writerow(data)


def save_json(data, path=None):
    """
    Saves data to a JSON file

    :param data: Data to save
    :type data: dict
    :param path: Save path, defaults to None (opens dialog)
    :type path: str, optional
    :raises ValueError: If data validation fails
    """
    root = Tk()
    root.withdraw()
    if path is None:
        try:
            path = filedialog.asks极好asfilename(defaultextension=".json", filetypes=[("json file", "*.json")],
                                                 title="Select file save path")
        except Exception as e:
            print(e)
    if path:
        if check_data_dict(data):
            data = save_prepare(data)
            with open(path, 'w') as json_file:
                json.dump(data, json_file, indent=2)


def save_metadata(data, path=None):
    """
    Saves metadata to a JSON file

    :param data: Metadata to save
    :type data: dict
    :param path: Save path, defaults to None (opens dialog)
    :type path: str, optional
    """
    root = Tk()
    root.withdraw()
    if path is None:
        try:
            path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("json file", "*.json")],
                                                title="Select file save path")
        except Exception as e:
            print(e)
    if path:
        data = save_prepare(data)
        with open(path, 'w') as json_file:
            json.dump(data, json_file, indent=2)


def save_txt(data, path=None):
    """
    Saves data to a text file

    :param data: Data to save
    :type data: dict
    :param path: Save path, defaults to None (opens dialog)
    :type path: str, optional
    :raises ValueError: If data validation fails
    """
    root = Tk()
    root.withdraw()
    if path is None:
        try:
            path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("txt file", "*.txt")],
                                                title="Select file save path")
        except Exception as e:
            print(e)
    if path:
        if check_data_dict(data):
            data = save_prepare(data)
            with open(path, 'w') as txt_file:
                json.dump(data, txt_file, indent=2)


def save_npy(data, path=None):
    """
    Saves data to a NumPy .npy file

    :param data: Data to save
    :type data: any
    :param path: Save path, defaults to None (opens dialog)
    :type path: str, optional
    """
    root = Tk()
    root.withdraw()
    if path is None:
        try:
            path = filedialog.asksaveasfilename(defaultextension=".npy", filetypes=[("npy files", "*.npy")],
                                                title="Select file save path")
        except Exception as e:
            print(e)
    if path:
        data = save_prepare(data)
        np.save(path, data)


def save_mat(data, path=None):
    """
    Saves data to a MATLAB .mat file

    :param data: Data to save
    :type data: dict
    :param path: Save path, defaults to None (opens dialog)
    :type path: str, optional
    """
    root = Tk()
    root.withdraw()
    if path is None:
        try:
            path = filedialog.asksaveasfilename(defaultextension=".mat", filetypes=[("mat files", "*.mat")],
                                                title="Select file save path")
        except Exception as e:
            print(e)
    if path:
        data = save_prepare(data)
        savemat(path, data)


def create_standard_bdf_file(file_name, n_channels, signals, channel_names,
                             physical_mins=None, physical_maxs=None,
                             digital_mins=None, digital_maxs=None,
                             sampling_frequency=256, technician='', equipment='',
                             patient_code='', patient_name='', birthdate=None,
                             admin_code='', annotations=None, transducers=None,
                             physical_dimensions=None):
    """
    Creates a standard BDF file from provided data

    :param file_name: Output filename
    :type file_name: str
    :param n_channels: Number of channels
    :type n_channels: int
    :param signals: Signal data (n_channels x n_samples)
    :type signals: np.ndarray
    :param channel_names: List of channel names
    :type channel_names: list[str]
    :param physical_mins: Physical minimum values, defaults to [-10000]*n_channels
    :type physical_mins: list[float], optional
    :param physical_maxs: Physical maximum values, defaults to [10000]*n_channels
    :type physical_maxs: list[float], optional
    :param digital_mins: Digital minimum values, defaults to [-32768]*n_channels
    :type digital_mins: list[int], optional
    :param digital_maxs: Digital maximum values, defaults to [32767]*n_channels
    :type digital_maxs: list[int], optional
    :param sampling_frequency: Sampling rate, defaults to 256
    :type sampling_frequency: float, optional
    :param technician: Technician name, defaults to ''
    :type technician: str, optional
    :param equipment: Equipment description, defaults to ''
    :type equipment: str, optional
    :param patient_code: Patient code, defaults to ''
    :type patient_code: str, optional
    :param patient_name: Patient name, defaults to ''
    :type patient_name: str, optional
    :param birthdate: Patient birthdate, defaults to None
    :type birthdate: str, optional
    :param admin_code: Administration code, defaults to ''
    :type admin_code: str, optional
    :param annotations: Event annotations [(onset, duration, description)], defaults to None
    :type annotations: list[tuple], optional
    :param transducers: Transducer types, defaults to ['EEG Electrode']*n_channels
    :type transducers: list[str], optional
    :param physical_dimensions: Physical dimensions, defaults to ['uV']*n_channels
    :type physical_dimensions: list[str], optional
    """
    if physical_mins is None:
        physical_mins = [-10000] * n_channels
    if physical_maxs is None:
        physical_maxs = [10000] * n_channels

    if digital_mins is None:
        digital_mins = [-32768] * n_channels
    if digital_maxs is None:
        digital_maxs = [32767] * n_channels

    f = pyedflib.EdfWriter(file_name, n_channels, file_type=pyedflib.FILETYPE_BDFPLUS)
    current_time = datetime.now().strftime("%d %b %Y %H:%M:%S")
    start_datetime = datetime.strptime(current_time, "%d %b %Y %H:%M:%S")
    f.setStartdatetime(start_datetime)
    f.setTechnician(technician)
    f.setEquipment(equipment)
    f.setPatientCode(patient_code)
    f.setPatientName(patient_name)
    if birthdate is not None:
        f.setBirthdate(birthdate)
    f.setAdmincode(admin_code)

    if transducers is None:
        transducers = ['EEG Electrode'] * n_channels
    if physical_dimensions is None:
        physical_dimensions = ['uV'] * n_channels

    for i in range(n_channels):
        f.setLabel(i, channel_names[i])
        f.setPhysicalMinimum(i, physical_mins[i])
        f.setPhysicalMaximum(i, physical_maxs[i])
        f.setDigitalMinimum(i, digital_mins[i])
        f.setDigitalMaximum(i, digital_maxs[i])
        f.setSamplefrequency(i, sampling_frequency)
        f.setTransducer(i, transducers[i])
        f.setPhysicalDimension(i, physical_dimensions[i])

    if annotations is not None:
        for onset, duration, description in annotations:
            f.writeAnnotation(onset, duration, str(description))

    f.writeSamples(signals)
    f.close()


if __name__ == '__main__':
    f = read_file()
    print(f['loc'])
    print(f['ch_names'])