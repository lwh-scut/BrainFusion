import json
import os
import sys

import matplotlib
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PyQt5.QtCore import Qt, QEvent, pyqtSignal, QAbstractTableModel
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QFileDialog, QDialog, QSizePolicy, QVBoxLayout, QPushButton, QHBoxLayout, QCheckBox, \
    QLineEdit, QLabel, QWidget, QMainWindow, QFormLayout, QApplication, QMessageBox, QListWidget, QListWidgetItem, \
    QDialogButtonBox, QScrollArea, QTableView
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.stats import ttest_ind
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import GridSearchCV

from BrainFusion.io.File_IO import read_neuracle_bdf, read_edf, read_csv, read_txt, read_npy, read_mat, read_json, \
    read_file_by_qt
from scipy import signal
from scipy.signal import iirnotch

from BrainFusion.utils.channels import drop_channels
from BrainFusion.utils.normalize import min_max_scaling_to_range, min_max_scaling_by_arrays
from BrainFusion.viewer.viewer_curve import RawCurvePlotDialog
from UI.ui_component import BFPushButton
import seaborn as sns

matplotlib.use('QtAgg')

import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QSizePolicy)

from pyqtgraph import PlotWidget, ScatterPlotItem, mkPen, InfiniteLine, TextItem
from PyQt5.QtWidgets import QVBoxLayout, QSizePolicy


class MatplotlibWidget(QWidget):
    """Widget for displaying a Matplotlib plot."""

    def __init__(self, parent=None):
        super(MatplotlibWidget, self).__init__(parent)
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        # self.plot_default()
        self.plot_data()

    def plot_default(self):
        """Default boxplot."""
        data = [np.random.randn(100) for _ in range(5)]
        self.plot(data, title="Default Boxplot", y_label="Value", x_label="Group", x_ticks=["A", "B", "C", "D", "E"])

    def plot(self, data, title="", y_label="", x_label="", legend=None, x_ticks=None, y_range=None, color="blue"):
        self.ax.clear()
        self.ax.boxplot(data, patch_artist=True, boxprops=dict(facecolor=color))
        self.ax.set_title(title)
        self.ax.set_ylabel(y_label)
        self.ax.set_xlabel(x_label)

        if x_ticks:
            self.ax.set_xticklabels(x_ticks)
        if y_range:
            self.ax.set_ylim(y_range)
        if legend:
            self.ax.legend(legend)

        self.canvas.draw()

    def plot_data(self):
        # 定义EEG通道和fNIRS通道选择
        eeg_channel_select = [['FCC5h'], ['FCC3h'], ['FCC4h'], ['FCC6h'],
                              ['CCP5h'], ['CCP3h'], ['CCP4h'], ['CCP6h']]
        fnirs_channel_select = [['S8_D9', 'S8_D10', 'S7_D10', 'S7_D9'],
                                ['S8_D11', 'S10_D11', 'S10_D10', 'S8_D10'],
                                ['S12_D13', 'S12_D15', 'S11_D15', 'S11_D13'],
                                ['S12_D16', 'S14_D16', 'S14_D15', 'S12_D15'],
                                ['S7_D10', 'S9_D10', 'S9_D5', 'S7_D5'],
                                ['S10_D10', 'S10_D12', 'S9_D12', 'S9_D10'],
                                ['S11_D15', 'S13_D15', 'S13_D14', 'S11_D14'],
                                ['S14_D15', 'S14_D8', 'S13_D8', 'S13_D15']]

        # NVC结果文件路径
        nvc_output_folder = 'E:\\DATA\\公开数据集\\EEG-fNIRS\\TUBerlinBCI\\Analysis Folder\\NVC\\02'
        json_output_file = os.path.join(nvc_output_folder, 'nvc_results.json')

        # 加载NVC结果
        with open(json_output_file, 'r') as json_file:
            nvc_results = json.load(json_file)

        # 仅分析subject 12
        subject = 'subject 15'
        subject_data = nvc_results['data'].get(subject, [])

        # 初始化保存左右手运动想象的NVC值
        nvc_left = {eeg_ch[0]: [] for eeg_ch in eeg_channel_select}
        nvc_right = {eeg_ch[0]: [] for eeg_ch in eeg_channel_select}

        # 遍历被试的数据
        for epoch in subject_data:
            # 根据标签确定是左手还是右手
            label = nvc_results['Labels'][subject][subject_data.index(epoch)]

            # 遍历每个EEG通道
            for eeg_idx, eeg_ch_list in enumerate(eeg_channel_select):
                eeg_ch = eeg_ch_list[0]  # 当前EEG通道

                # 计算对应四个fNIRS通道的NVC均值
                fnirs_channels = [ch + ' hbo' for ch in fnirs_channel_select[eeg_idx]]
                nvc_values = []

                for result in epoch:
                    if result['EEG_Channel'] == eeg_ch and result['fNIRS_Channel'] in fnirs_channels:
                        # 取NVC的绝对值
                        nvc_values.append(abs(result['NVC_Value']))

                # 保存NVC值
                if nvc_values:
                    nvc_mean = np.mean(nvc_values)
                    if label == 'left':
                        nvc_left[eeg_ch].append(nvc_mean)
                    elif label == 'right':
                        nvc_right[eeg_ch].append(nvc_mean)

        eeg_channels = list(nvc_left.keys())
        self.ax.clear()

        # 绘制左右手的箱线图
        boxplot_data_left = [nvc_left[ch] for ch in eeg_channels]
        boxplot_data_right = [nvc_right[ch] for ch in eeg_channels]

        # 左手箱线图
        positions_left = np.arange(len(eeg_channels)) * 2.0
        self.ax.boxplot(boxplot_data_left, positions=positions_left, widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor="skyblue"), labels=eeg_channels)

        # 右手箱线图
        positions_right = positions_left + 0.8
        self.ax.boxplot(boxplot_data_right, positions=positions_right, widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor="salmon"))

        # 添加显著性标记
        significance_level = 0.05
        bracket_height = 0.05  # 括号的高度
        asterisk_offset = 0.01  # 星号相对于括号的偏移
        line_width = 1.5  # 横线宽度

        for i, ch in enumerate(eeg_channels):
            # 进行 t 检验，比较左手和右手的 NVC 值
            t_stat, p_value = ttest_ind(nvc_left[ch], nvc_right[ch], nan_policy='omit')

            # 如果 p 值小于显著性水平，则标记星号和括号
            if p_value < significance_level:
                max_value = max(max(nvc_left[ch]), max(nvc_right[ch]))  # 获取最大值用于放置星号和括号
                y_bracket = max_value + bracket_height  # 括号位置
                y_asterisk = y_bracket + asterisk_offset  # 星号位置

                # 在两个箱线图之间绘制星号
                self.ax.text(positions_left[i] + 0.4, y_asterisk, '*', ha='center', va='bottom', fontsize=14,
                             color='red')

                # 在星号下方绘制横线括号
                self.ax.plot([positions_left[i], positions_right[i]], [y_bracket, y_bracket], color='black',
                             lw=line_width)
                self.ax.plot([positions_left[i], positions_left[i]], [y_bracket, y_bracket - 0.02], color='black',
                             lw=line_width)
                self.ax.plot([positions_right[i], positions_right[i]], [y_bracket, y_bracket - 0.02], color='black',
                             lw=line_width)

        # 添加标签和图例
        self.ax.set_xlabel('EEG Channels', fontsize=12)
        self.ax.set_ylabel('Absolute NVC Value', fontsize=12)
        self.ax.set_title(f'Neurovascular Coupling (NVC) Boxplot for Left and Right Hand Motor Imagery',
                          fontsize=12)

        # 调整 X 轴标签位置
        self.ax.set_xticks(positions_left + 0.4)
        self.ax.set_xticklabels(eeg_channels)

        # 添加图例
        legend_elements = [plt.Line2D([0], [0], color="skyblue", lw=4, label='Left Hand'),
                           plt.Line2D([0], [0], color="salmon", lw=4, label='Right Hand'),
                           plt.Line2D([0], [0], marker='*', color='w', label='p<0.05',
                                      markerfacecolor='red', markersize=10)]

        self.ax.legend(handles=legend_elements, loc='upper right')

        self.canvas.draw()


class TestBoxPlot(QDialog):
    def __init__(self, parent=None):
        super(TestBoxPlot, self).__init__(parent)
        self.matplotlib_widget = MatplotlibWidget()
        self.vlayout = QVBoxLayout(self)

        # figure config
        self.bnt_save = BFPushButton('Save')
        self.bnt_save.setFixedWidth(150)
        # self.bnt_save.setFixedHeight(60)
        self.bnt_setting = BFPushButton('Settings')
        self.bnt_setting.setFixedWidth(150)
        self.bnt_layout = QHBoxLayout()
        self.bnt_layout.addStretch(1)
        self.bnt_layout.addWidget(self.bnt_save)
        self.bnt_layout.addWidget(self.bnt_setting)

        self.vlayout.addLayout(self.bnt_layout)
        self.vlayout.addWidget(self.matplotlib_widget)


class MplCanvas(FigureCanvas):
    """自定义Matplotlib画布类，用于嵌入PyQt5"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class TestMLPlot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 1200, 600)

        # 创建主窗口的widget
        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QVBoxLayout(widget)
        self.bnt_save = BFPushButton('Save')
        self.bnt_save.setFixedWidth(150)
        # self.bnt_save.setFixedHeight(60)
        self.bnt_setting = BFPushButton('Settings')
        self.bnt_setting.setFixedWidth(150)
        self.bnt_layout = QHBoxLayout()
        self.bnt_layout.addStretch(1)
        self.bnt_layout.addWidget(self.bnt_save)
        self.bnt_layout.addWidget(self.bnt_setting)
        layout.addLayout(self.bnt_layout)
        figure_layout = QHBoxLayout()

        # 创建 ROC 曲线和混淆矩阵的画布
        self.roc_canvas = MplCanvas(self, width=5, height=4)
        self.cm_canvas = MplCanvas(self, width=5, height=4)

        figure_layout.addWidget(self.roc_canvas)
        figure_layout.addWidget(self.cm_canvas)

        layout.addLayout(figure_layout)

        # 执行 SVM 训练和可视化
        self.run_svm_and_plot()

    def run_svm_and_plot(self):
        # 生成假数据：1000个样本，20个特征，2个类别
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

        # SVM 模型和参数范围
        svm_model = svm.SVC(probability=True)
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf']
        }

        # 使用网格搜索进行训练
        grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='roc_auc')
        grid_search.fit(X, y)

        # 获取最佳模型
        best_model = grid_search.best_estimator_

        y_pre = best_model.predict(X)

        # 生成预测分数
        y_scores = best_model.predict_proba(X)[:, 1]

        # 计算 ROC 曲线
        fpr, tpr, _ = roc_curve(y, y_scores)
        roc_auc = auc(fpr, tpr)

        # 绘制 ROC 曲线
        self.roc_canvas.axes.clear()
        self.roc_canvas.axes.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        self.roc_canvas.axes.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        self.roc_canvas.axes.set_xlim([0.0, 1.0])
        self.roc_canvas.axes.set_ylim([0.0, 1.05])
        self.roc_canvas.axes.set_xlabel('False Positive Rate')
        self.roc_canvas.axes.set_ylabel('True Positive Rate')
        self.roc_canvas.axes.set_title('Receiver Operating Characteristic (ROC)')
        self.roc_canvas.axes.legend(loc="lower right")
        self.roc_canvas.draw()

        # 混淆矩阵
        cm = confusion_matrix(y, y_pre)

        # 绘制混淆矩阵
        self.cm_canvas.axes.clear()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=self.cm_canvas.axes)
        self.cm_canvas.axes.set_title('Confusion Matrix')
        self.cm_canvas.axes.set_xlabel('Predicted Label')
        self.cm_canvas.axes.set_ylabel('True Label')
        self.cm_canvas.draw()


import os
import sys
from PyQt5 import QtWidgets
import mne
from mne.datasets import sample
import sys
import pyvista as pv
from PyQt5 import QtWidgets
from mne.viz import create_3d_figure, set_3d_backend
from pyvistaqt import QtInteractor, BackgroundPlotter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 示例调用
import sys
import pyvista as pv
from PyQt5 import QtWidgets
from pyvistaqt import QtInteractor

set_3d_backend('pyvistaqt')


class Sensors3D(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # 设置窗口标题和尺寸
        self.setGeometry(100, 100, 800, 600)

        # 创建一个中央widget并设置为布局
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)

        # 创建布局
        self.layout = QtWidgets.QVBoxLayout()
        central_widget.setLayout(self.layout)
        self.plotter = BackgroundPlotter(show=False)

        self.figure = create_3d_figure((800, 600))
        self.figure._plotter = self.plotter

        self.layout.addWidget(self.plotter.interactor)

class TestEEGPlot(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 1200, 600)
        self.layout = QVBoxLayout(self)
        self.bnt_save = BFPushButton('Save')
        self.bnt_save.setFixedWidth(150)
        # self.bnt_save.setFixedHeight(60)
        self.bnt_setting = BFPushButton('Settings')
        self.bnt_setting.setFixedWidth(150)
        self.bnt_sensor = BFPushButton('Plot Sensors')
        self.bnt_sensor.setFixedWidth(150)
        self.bnt_layout = QHBoxLayout()
        self.bnt_layout.addStretch(1)
        self.bnt_layout.addWidget(self.bnt_sensor)
        self.bnt_layout.addWidget(self.bnt_save)
        self.bnt_layout.addWidget(self.bnt_setting)

        self.layout.addLayout(self.bnt_layout)
        data, file_path = read_file_by_qt(self)

        print(data)
        print(file_path)
        self.drawing_widget = RawCurvePlotDialog(data=data, filePath=file_path[0])
        self.eeg_sensor = TestEEGSensorPlot()
        self.layout.addWidget(self.drawing_widget)

        self.bnt_sensor.clicked.connect(lambda :self.eeg_sensor.show())

        self.drawing_widget.plot_data(self.drawing_widget.current_page)


class TestEEGSensorPlot(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 1200, 600)
        self.layout = QVBoxLayout(self)
        sensors = Sensors3D()
        self.bnt_save = BFPushButton('Save')
        self.bnt_save.setFixedWidth(150)
        # self.bnt_save.setFixedHeight(60)
        self.bnt_layout = QHBoxLayout()
        self.bnt_layout.addStretch(1)
        self.bnt_layout.addWidget(self.bnt_save)
        self.layout.addLayout(self.bnt_layout)
        self.layout.addWidget(sensors)


        eeg_path = 'C:\\Users\\28164\\Desktop\\test\\open_dataset\\eeg\\struct_1.bdf'
        fnirs_path = 'C:\\Users\\28164\\Desktop\\test\\open_dataset\\struct_1.snirf'

        raw_nirs = mne.io.read_raw_snirf(fnirs_path, preload=True)
        raw_eeg = mne.io.read_raw_bdf(eeg_path, preload=True)
        events, _ = mne.events_from_annotations(raw_eeg)
        raw_eeg.set_channel_types({"VEOG": "eog", "HEOG": "eog"})
        raw_eeg.set_montage('standard_1005')

        subjects_dir = os.path.join(mne.datasets.sample.data_path(), "subjects")
        brain_eeg_fnirs = mne.viz.Brain(
            "fsaverage", subjects_dir=subjects_dir, background="w", cortex="0.5", alpha=0.3,
            title='EEG and fNIRS Sensors', show=False, figure=sensors.figure
        )
        # brain_eeg_fnirs.add_sensors(
        #     raw_nirs.info,
        #     trans="fsaverage",
        #     # fnirs=["channels", "pairs", "sources", "detectors"],
        #     fnirs=["pairs", "sources", "detectors"],
        #
        # )
        brain_eeg_fnirs.add_sensors(
            raw_eeg.info,
            trans="fsaverage",

        )
        brain_eeg_fnirs.show_view(azimuth=0, elevation=0, distance=500)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TestBoxPlot()
    window.show()
    sys.exit(app.exec_())
