# -*- coding: utf-8 -*-
# @Time    : 2024/5/28 15:44
# @Author  : XXX
# @Site    : 
# @File    : normalize.py
# @Software: PyCharm 
# @Comment :
import numpy as np


def normalize(signal):
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))


def min_max_scaling_to_range(array, new_min=-1, new_max=1):
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = new_min + (new_max - new_min) * (array - min_val) / (max_val - min_val)
    return normalized_array


def min_max_scaling_by_arrays(arrays, new_min=-1, new_max=1):
    normalized_arrays = []
    for array in arrays:
        normalized_array = min_max_scaling_to_range(array, new_min, new_max)
        normalized_arrays.append(normalized_array)
    return np.array(normalized_arrays)


def slice_by_minlength(a, b):
    if len(a) > len(b):
        return a[:len(b)], b
    else:
        return a, b[:len(a)]


def convert_to_serializable(obj):
    """递归地将 numpy 类型转换为原生 Python 类型"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj
