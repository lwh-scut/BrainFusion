# -*- coding: utf-8 -*-
# @Time    : 2024/5/28 15:44
# @Author  : XXX
# @Site    : 
# @File    : normalize.py
# @Software: PyCharm 
# @Comment :
import numpy as np


def normalize(signal):
    """
    Normalize signal values to 0-1 range.

    :param signal: Input signal values
    :type signal: np.ndarray
    :return: Normalized signal
    :rtype: np.ndarray
    """
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))


def min_max_scaling_to_range(array, new_min=-1, new_max=1):
    """
    Scale array values to specified range.

    :param array: Input numerical array
    :type array: np.ndarray
    :param new_min: Minimum value of target range
    :type new_min: float
    :param new_max: Maximum value of target range
    :type new_max: float
    :return: Array scaled to new range
    :rtype: np.ndarray
    """
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = new_min + (new_max - new_min) * (array - min_val) / (max_val - min_val)
    return normalized_array


def min_max_scaling_by_arrays(arrays, new_min=-1, new_max=1):
    """
    Apply min-max scaling to list of arrays.

    :param arrays: List of input arrays
    :type arrays: list
    :param new_min: Minimum value of target range
    :type new_min: float
    :param new_max: Maximum value of target range
    :type new_max: float
    :return: List of scaled arrays
    :rtype: list
    """
    normalized_arrays = []
    for array in arrays:
        normalized_array = min_max_scaling_to_range(array, new_min, new_max)
        normalized_arrays.append(normalized_array)
    return np.array(normalized_arrays)


def slice_by_minlength(a, b):
    """
    Trim arrays to same length based on shorter array.

    :param a: First input array
    :type a: np.ndarray or list
    :param b: Second input array
    :type b: np.ndarray or list
    :return: Equal-length sliced arrays
    :rtype: tuple
    """
    # Determine which array is shorter
    if len(a) > len(b):
        return a[:len(b)], b
    else:
        return a, b[:len(a)]


def convert_to_serializable(obj):
    """
    Recursively convert numpy types to Python serializable types.

    :param obj: Input object to convert
    :type obj: any
    :return: Serializable Python object
    :rtype: native Python type
    """
    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle numpy scalars
    elif isinstance(obj, np.generic):
        return obj.item()
    # Handle dictionaries
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    # Handle lists
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    # Return other types unchanged
    return obj