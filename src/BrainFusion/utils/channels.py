# -*- coding: utf-8 -*-
# @Time    : 2024/5/28 20:43
# @Author  : XXX
# @Site    : 
# @File    : channels.py
# @Software: PyCharm 
# @Comment :
import os

import numpy as np


def drop_channels(raw_data, channels, bad_channels):
    """
    Remove specified bad channels from raw EEG/fNIRS data.

    :param raw_data: 2D array of signal data (channels × samples)
    :type raw_data: np.ndarray
    :param channels: List of channel identifiers
    :type channels: list
    :param bad_channels: Channels to exclude
    :type bad_channels: list
    :return: Cleaned data array and channel list
    :rtype: tuple(np.ndarray, list)
    """
    # Convert to numpy array if needed
    raw_data = np.array(raw_data)

    # Calculate indices for good channels
    keep_indices = [i for i, ch in enumerate(channels) if ch not in bad_channels]

    # Filter data and channels
    cleaned_data = raw_data[keep_indices, :]
    cleaned_channels = [ch for i, ch in enumerate(channels) if ch not in bad_channels]

    return cleaned_data, cleaned_channels


def is_multidimensional_list(lst):
    """
    Determine if a list contains nested list structures.

    :param lst: Candidate input list
    :type lst: any
    :return: True if multidimensional, False otherwise
    :rtype: bool
    """
    # Check if input is a list
    if not isinstance(lst, list):
        return False
    # Check for nested lists
    return any(isinstance(i, list) for i in lst)


def convert_ndarray_to_list(item):
    """
    Recursively convert numpy ndarrays to standard Python lists.

    :param item: Input numpy array or nested structure
    :type item: any
    :return: Python list structure
    :rtype: list or scalar
    """
    # Convert ndarrays
    if isinstance(item, np.ndarray):
        return convert_ndarray_to_list(item.tolist())
    # Process nested lists
    elif isinstance(item, list):
        return [convert_ndarray_to_list(sub) for sub in item]
    # Return scalar values unchanged
    return item