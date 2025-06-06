# -*- coding: utf-8 -*-
# @Time    : 2024/3/1 14:16
# @Author  : Li WenHao
# @Site    : South China University of Technology
# @File    : plot_raw.py
# @Software: PyCharm 
# @Comment :
from tkinter import Tk, filedialog

import matplotlib
import mne
import numpy as np
import matplotlib.pyplot as plt

from BrainFusion.io.File_IO import read_neuracle_bdf, read_edf, read_csv, read_txt, read_npy, read_mat, read_json, \
    read_file, read_file_by_qt
from BrainFusion.viewer.plot_with_dialog import min_max_scaling_to_range, min_max_scaling_by_arrays

matplotlib.use('QtAgg')


def plot_raw_by_file(widget, path=None):
    data, _ = read_file_by_qt(widget, path)
    if data:
        plot_raw(data=data['data'], channel=data['ch_names'])


def plot_raw(data, channel=None, sharey=False, line_color='black', linewidth=0.5, is_save=False, save_path=None):
    # 判断数据类型
    if isinstance(data, np.ndarray):
        # 获取数据维度
        dimensions = data.ndim
    elif isinstance(data, list):
        # 获取数据维度
        dimensions = len(np.array(data).shape)
    else:
        print("Unsupported data type.")
        return None
    if dimensions == 1:
        length = np.array(data).shape[0]
        num_channels = 1
        if channel is None:
            channel = ['channel']
        fig, axes = plt.subplots(num_channels, 1, figsize=(10, 6), sharex=True, sharey=sharey)
        axes.plot(data, color=line_color, linewidth=linewidth)
        axes.set_ylabel(f' {channel[0]}', rotation=0, ha='right')
        axes.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False,
                         right=False,
                         labelleft=False)
        axes.spines['top'].set_color('lightgrey')  # 设置坐标轴边框颜色
        axes.spines['bottom'].set_color('lightgrey')
        axes.spines['right'].set_color('lightgrey')
        axes.spines['left'].set_color('lightgrey')
        # 调整曲线的起点离y轴的距离
        axes.set_xlim(left=-10, right=length)
        fig.subplots_adjust(hspace=0, wspace=0, bottom=0.02, left=0.1, top=0.98, right=0.98)  # 去除子图间的垂直间隙
        # plt.show()
        if is_save:
            plt.savefig(save_path, dpi=300)

    elif dimensions == 2:
        data = np.array(data)
        length = data.shape[1]
        num_channels = data.shape[0]
        if channel is None:
            channel = [str(i) for i in range(1, num_channels + 1)]
        fig, axes = plt.subplots(num_channels, 1, figsize=(10, 6), sharex=True, sharey=sharey)
        for i in range(num_channels):
            axes[i].plot(data[i, :30000], color=line_color, linewidth=linewidth)
            axes[i].set_ylabel(f' {channel[i]}', rotation=0, ha='right')
            axes[i].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False,
                                right=False,
                                labelleft=False)
            axes[i].spines['top'].set_color('lightgrey')  # 设置坐标轴边框颜色
            axes[i].spines['bottom'].set_color('lightgrey')
            axes[i].spines['right'].set_color('lightgrey')
            axes[i].spines['left'].set_color('lightgrey')
            # 调整曲线的起点离y轴的距离
            axes[i].set_xlim(left=-10)
        # 添加时间轴
        # 生成时间轴数据
        axes[-1].tick_params(axis='both', which='both', bottom=True, top=False, left=False, right=False,
                             labelbottom=True)
        axes[-1].spines['top'].set_color('lightgrey')
        axes[-1].spines['bottom'].set_color('lightgrey')
        axes[-1].spines['right'].set_color('lightgrey')
        axes[-1].spines['left'].set_color('lightgrey')
        fig.subplots_adjust(hspace=0, wspace=0, bottom=0.05, left=0.1, top=0.98, right=0.98)  # 去除子图间的垂直间隙
        # plt.show()
        if is_save:
            plt.savefig(save_path, dpi=300)
    else:
        print(f"Data has {dimensions} dimensions.Not support.")
        return None


def plot_eeg_psd(data, is_relative=False, is_norm=True, title=''):
    title_list = None
    num_fig = np.array(data['data']).shape[1]
    fig, axes = plt.subplots(1, num_fig, figsize=(8, 5), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0, wspace=0.05, bottom=0.08, left=0.05, top=0.98, right=0.98)  # 去除子图间的垂直间隙
    title_list = ['Δ wave band', 'θ wave band', 'α wave band', 'β wave band', 'γ wave band']
    fig.suptitle(f"{title} EEG Power Spectral Density")

    if data:
        # self.fig.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        montage = mne.channels.make_standard_montage('standard_1020')
        info = mne.create_info(ch_names=data['ch_names'], sfreq=data['srate'], ch_types='eeg')
        if is_norm:
            if is_relative:
                norm_data = min_max_scaling_to_range(np.array(data['data']).T)
                data_range = (-1, 1)
            else:
                norm_data = min_max_scaling_by_arrays(np.array(data['data']).T)
                data_range = (-1, 1)
        else:
            norm_data = np.array(data['data']).T
            data_range = (-1, 1)
        for i, psd in enumerate(norm_data):
            evoked = mne.EvokedArray(data=np.array(data['data']), info=info)
            evoked.set_montage(montage)
            axes[i].clear()
            if title_list:
                axes[i].set_title(title_list[i])
            mne.viz.plot_topomap(psd, evoked.info,
                                 axes=axes[i], show=False
                                 , sensors=True, vlim=data_range)
            axes[i].figure.canvas.draw()
