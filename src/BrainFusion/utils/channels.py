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
    Remove specified bad channels from the raw data.

    Parameters:
    - raw_data: 2D numpy array where each row corresponds to a channel's signal data.
    - channels: List of channel names corresponding to the rows in raw_data.
    - bad_channels: List of channel names to be removed.

    Returns:
    - cleaned_data: 2D numpy array with the bad channels removed.
    - cleaned_channels: List of channel names with the bad channels removed.
    """
    # Ensure raw_data is a numpy array
    raw_data = np.array(raw_data)

    # Find the indices of channels to keep
    indices_to_keep = [i for i, ch in enumerate(channels) if ch not in bad_channels]

    # Extract the cleaned data and channels
    cleaned_data = raw_data[indices_to_keep, :]
    cleaned_channels = [ch for i, ch in enumerate(channels) if ch not in bad_channels]

    return cleaned_data, cleaned_channels


def is_multidimensional_list(lst):
    # 如果不是列表，返回False
    if not isinstance(lst, list):
        return False
    # 如果列表中的任何一个元素是列表，则返回True
    return any(isinstance(i, list) for i in lst)


def convert_ndarray_to_list(item):
    if isinstance(item, np.ndarray):
        # Convert the numpy array to a list and continue processing
        return convert_ndarray_to_list(item.tolist())
    elif isinstance(item, list):
        # Recursively convert each element in the list
        return [convert_ndarray_to_list(sub_item) for sub_item in item]
    else:
        # If the item is neither a numpy array nor a list, return it as is
        return item
