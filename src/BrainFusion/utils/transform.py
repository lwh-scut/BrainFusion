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
    将包含光源和探测器信息的字典转换为 mne.io.Raw 对象。

    :param data_dict: 包含 SNIRF 文件结构信息的字典。
    :return: mne.io.Raw 对象。
    """

    ch_names = list(data_dict['ch_names'])  # 通道名称
    data = data_dict['data']  # 数据矩阵
    times = data_dict['time']  # 时间信息
    sfreq = 1.0 / np.mean(np.diff(times))  # 采样率从时间间隔中计算

    source_pos_3d = data_dict['loc']['sourcePos3D']
    detector_pos_3d = data_dict['loc']['detectorPos3D']

    info = mne.create_info(ch_names, sfreq, ch_types='fnirs_cw_amplitude')  # 创建原始信息结构，类型设为 EEG
    raw = mne.io.RawArray(data.T, info)  # 数据矩阵需要转置为 (n_channels, n_timepoints)

    for i, ch_name in enumerate(ch_names):
        match = re.match(r"S(\d+)_D(\d+)\s+(\d+)", ch_name)
        if match:
            source_idx = int(match.group(1))  # 源号
            detector_idx = int(match.group(2))  # 探测器号
            wavelength = int(match.group(3))  # 波长
        loc = np.zeros(12)
        loc[3:6] = source_pos_3d[source_idx - 1] / 100  # 设置光源位置
        loc[6:9] = detector_pos_3d[detector_idx - 1] / 100  # 设置探测器位置
        loc[0:3] = (loc[3:6] + loc[6:9]) / 2
        loc[9] = wavelength  # 设置波长
        raw.info['chs'][i]['loc'] = loc  # 将位置信息赋给通道信息
        raw.info['chs'][i]['coord_frame'] = FIFF.FIFFV_COORD_UNKNOWN

    events = data_dict.get('events', [])
    for event in events:
        onset, duration, label = event
        raw.annotations.append(onset, duration, label)

    return raw


def raw_to_dict(raw):
    """
    将 mne.io.Raw 对象转换为字典，适配 dict_to_snirf 函数。

    :param raw: mne.io.Raw 对象。
    :return: 包含 SNIRF 文件结构信息的字典。
    """
    # 获取通道信息
    ch_names = raw.info['ch_names']
    sfreq = raw.info['sfreq']
    data, times = raw.get_data(return_times=True)

    # 获取光源和探测器位置
    source_pos_3d = []
    detector_pos_3d = []
    sd_pairs = []  # 光源-探测器配对
    wavelengths = []

    for ch_idx, ch in enumerate(raw.info['chs']):
        loc = ch['loc']
        source_pos = loc[3:6]  # 光源位置
        detector_pos = loc[6:9]  # 探测器位置
        wavelength = loc[9]  # 波长信息

        # 添加到列表中
        if source_pos.tolist() not in source_pos_3d:
            source_pos_3d.append(source_pos.tolist())
        if detector_pos.tolist() not in detector_pos_3d:
            detector_pos_3d.append(detector_pos.tolist())
        if wavelength not in wavelengths:
            wavelengths.append(wavelength)

        # 配对索引
        source_idx = source_pos_3d.index(source_pos.tolist()) + 1
        detector_idx = detector_pos_3d.index(detector_pos.tolist()) + 1
        sd_pairs.append((source_idx, detector_idx))

    # 获取事件信息（如有）
    events = []
    if raw.annotations is not None:
        for annotation in raw.annotations:
            onset = annotation['onset']
            duration = annotation['duration']
            label = annotation['description']
            events.append([onset, duration, label])

    # 获取标志点位置
    landmark_pos_3d = []
    landmark_labels = []
    if 'dig' in raw.info and raw.info['dig'] is not None:
        for dig_point in raw.info['dig']:
            if dig_point['kind'] == mne.io.constants.FIFF.FIFFV_POINT_EEG:  # 光源或探测器
                landmark_pos_3d.append(dig_point['r'])
                landmark_labels.append(f"Point {len(landmark_labels) + 1}")
            elif dig_point['kind'] in (
                    mne.io.constants.FIFF.FIFFV_POINT_LPA,
                    mne.io.constants.FIFF.FIFFV_POINT_NASION,
                    mne.io.constants.FIFF.FIFFV_POINT_RPA,
            ):  # 标志点
                landmark_pos_3d.append(dig_point['r'])
                landmark_labels.append({
                                           mne.io.constants.FIFF.FIFFV_POINT_LPA: "LPA",
                                           mne.io.constants.FIFF.FIFFV_POINT_NASION: "Nasion",
                                           mne.io.constants.FIFF.FIFFV_POINT_RPA: "RPA",
                                       }[dig_point['kind']])

    # 构建字典
    data_dict = {
        'data': data,  # 转置为 (n_timepoints, n_channels)
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

    return data_dict


def raw_to_dict_eeg(raw):
    """
        将 mne.io.Raw 对象转换为字典，适配 dict_to_snirf 函数。

        :param raw: mne.io.Raw 对象。
        :return: 包含 SNIRF 文件结构信息的字典。
        """
    # 获取通道信息
    ch_names = raw.info['ch_names']
    sfreq = raw.info['sfreq']
    data, times = raw.get_data(return_times=True)

    # 获取事件信息（如有）
    events = []
    if raw.annotations is not None:
        for annotation in raw.annotations:
            onset = annotation['onset']
            duration = annotation['duration']
            label = annotation['description']
            events.append([onset, duration, label])

    # 构建字典
    data_dict = {
        'data': data,  # 转置为 (n_timepoints, n_channels)
        'time': times,
        'events': events,
        'ch_names': ch_names,
        'srate': sfreq,
    }

    return data_dict


def dict_to_info(data_dict, filePath):
    """
    将 data_dict 中除 'data' 外的内容提取，并保存到 JSON 文件中。
    Args:
        data_dict (dict): 包含数据和元信息的字典。
        filePath (str): 要保存的 JSON 文件路径。
    """
    # 创建 info_dict，只保留 data_dict 中除 'data' 外的键值对
    info_dict = {key: value for key, value in data_dict.items() if key not in ['data', 'time', 'events']}

    # 处理 numpy 数组为可序列化格式
    def serialize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # 将 numpy 数组转换为列表
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)  # 将 numpy float 转换为 Python float
        if isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)  # 将 numpy int 转换为 Python int
        raise TypeError(f"无法序列化对象类型: {type(obj)}")

    # 将 info_dict 中的 numpy 对象转换为可 JSON 序列化的格式
    serializable_info_dict = json.loads(json.dumps(info_dict, default=serialize))

    # 保存为 JSON 文件
    with open(filePath, 'w', encoding='utf-8') as f:
        json.dump(serializable_info_dict, f, ensure_ascii=False, indent=4)


def read_info(filePath):
    """
    从 JSON 文件中读取数据并转换为字典。
    Args:
        filePath (str): JSON 文件路径。
    Returns:
        dict: 从 JSON 文件中读取的数据字典。
    """
    with open(filePath, 'r', encoding='utf-8') as f:
        info_dict = json.load(f)

    return info_dict


if __name__ == '__main__':
    # bdf_path = 'E:\\DATA\\minilab数据\\音乐实验\\Musical Imagery\\20241113-打标点整理后\\bdf\\2001.bdf'
    # bdf_output_path = 'C:\\Users\\28164\\Desktop\\test\\bdf\\2001.bdf'
    #
    # bdf_dict = read_bdf(bdf_path)
    # print(np.array(bdf_dict['data'])[0, 0:10])
    # print(bdf_dict['events'][0])
    #
    # dict_to_bdf(bdf_dict, bdf_output_path)
    #
    # output_bdf_dict = read_bdf(bdf_output_path, is_digital=False)
    #
    # print(np.array(output_bdf_dict['data'])[0, 0:10])
    # print(output_bdf_dict['events'][0])

    snirf_path = 'E:\\DATA\\minilab数据\\音乐实验\\Musical Imagery\\20241113-打标点整理后\\snirf\\2001.snirf'
    snirf_output_path = 'C:\\Users\\28164\\Desktop\\test\\bdf\\2001.snirf'
    info_path = 'C:\\Users\\28164\\Desktop\\test\\bdf\\2001_info.json'
    snirf_dict = read_snirf(snirf_path)
    #
    # raw2 = mne.io.read_raw_snirf(snirf_path)
    # print(raw2.info)
    # print('raw2', raw2.info['chs'][0])
    # raw2.plot_sensors()
    # raw = dict_to_snirf_raw(snirf_dict)
    # print(raw.info['chs'][0])
    # print(snirf_dict['loc']['sourcePos3D'][0])
    # print(snirf_dict['loc']['detectorPos3D'][0])
    # print(snirf_dict['loc']['landmarkPos3D'][0])
    # raw.plot_sensors()
    # plt.show()

    dict_to_info(snirf_dict, info_path)
    info_dict = read_info(info_path)
    print(info_dict)
