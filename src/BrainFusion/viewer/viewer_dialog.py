# -*- coding: utf-8 -*-
# @Time    : 2024/5/28 10:34
# @Author  : XXX
# @Site    : 
# @File    : viewer.py
# @Software: PyCharm 
# @Comment :
import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QLineEdit, QPushButton, QDialog, QVBoxLayout, QFrame, \
    QFileDialog, QListWidget, QListWidgetItem, QGroupBox, QTabWidget, QLabel, QAction, QMenu

from BrainFusion.io.File_IO import read_file_by_qt
from BrainFusion.viewer.viewer_curve import RawCurvePlotDialog, plot_raw_by_file, fNIRSPlotDialog
from BrainFusion.viewer.viewer_plot import BarPlotWidget, CurvePlotWidget, TablePlotWidget, TopoMapPlotWidget
from BrainFusion.viewer.viewer_test import TestBoxPlot, TestMLPlot, TestEEGSensorPlot, TestEEGPlot
from BrainFusion.viewer.viewer_time_frequency import TimeFrequencyViewer
from BrainFusion.viewer.viewer_topomap import TopomapViewer
from UI.ui_component import BFPushButton


class BrainFusionViewer(QWidget):
    def __init__(self):
        super(BrainFusionViewer, self).__init__()
        # 创建导入数据按钮和显示文件名的 QLineEdit
        self.setWindowTitle("BrainFusion Viewer")
        self.setGeometry(800, 600, 1200, 600)

        self.curve_type = ['eeg', 'eeg_preprocess', 'fnirs', 'fnirs_preprocess', 'emg', 'emg_preprocess', 'ecg',
                           'ecg_preprocess']
        self.topomap_type = ['eeg_psd', 'eeg_microstate']
        self.time_frequency_type = ['stft']

        self.import_button = BFPushButton("Select Folder")
        self.import_button.setFixedWidth(150)
        self.file_name_lineedit = QLineEdit()
        self.file_name_lineedit.setFixedWidth(150)
        self.file_name_lineedit.setReadOnly(True)  # 设置为只读，用户不能手动编辑

        # 连接按钮的点击事件
        self.groupbox_import = QGroupBox("Data Files")
        import_layout = QVBoxLayout(self.groupbox_import)
        self.import_button.clicked.connect(self.open_folder)

        # 创建布局并添加控件
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.import_button)
        top_layout.addWidget(self.file_name_lineedit)

        # left_layout.addLayout(top_layout)
        # 创建列表控件
        self.listWidget = QListWidget(self)
        self.listWidget.setFixedWidth(300)
        # 右键菜单支持
        self.listWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.listWidget.customContextMenuRequested.connect(self.show_context_menu)
        self.listWidget.doubleClicked.connect(self.list_item_double_clicked)

        import_layout.addLayout(top_layout)
        import_layout.addWidget(self.listWidget)

        self.groupbox_import.setFixedWidth(325)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.groupbox_import)
        # left_layout.addWidget(self.groupbox_events)

        # 创建frame
        self.drawing_frame = QFrame()
        self.drawing_frame.setFrameShape(QFrame.StyledPanel)  # 设置 QFrame 的样式
        self.drawing_frame.setStyleSheet("background-color: white;")
        self.drawing_layout = QVBoxLayout(self.drawing_frame)

        self.bnt_label = QLabel("Visualisation")
        self.bnt_label.setStyleSheet("font-family: 'Times New Roman'; font-size: 14pt; font-weight: bold;")
        self.bnt_layout = QHBoxLayout()
        self.bnt_layout.addWidget(self.bnt_label)
        self.bnt_layout.addStretch(1)
        self.drawing_layout.addLayout(self.bnt_layout)

        # self.drawing_layout
        self.drawing_widget = QTabWidget()
        self.drawing_widget.setTabsClosable(True)
        self.drawing_widget.tabCloseRequested.connect(self.close_tab)

        self.drawing_layout.addWidget(self.drawing_widget)

        # 创建主布局并添加之前的布局和控件
        layout = QHBoxLayout(self)
        layout.addLayout(left_layout)
        # layout.addStretch(1)
        layout.addWidget(self.drawing_frame)  # 将绘图用的 QFrame 添加到布局中

    def show_context_menu(self, position):
        """
        显示右键菜单，用于选择绘图类型。
        """
        # 获取当前选中的项目
        item = self.listWidget.itemAt(position)
        if not item:
            return
        file_path = item.data(32)  # 获取保存在自定义角色的完整路径
        data, file_path = read_file_by_qt(self, [file_path])
        name = item.text()

        # 创建右键菜单
        menu = QMenu(self)

        # 添加菜单选项
        curve_action = QAction("Line Chart", self)
        bar_action = QAction("Bar Chart", self)
        topo_action = QAction("Topographic Map", self)
        heatmap_action = QAction("热力图", self)
        network_action = QAction("网络图", self)
        table_action = QAction("Data Table", self)

        # data_type = data['type']
        # # 根据 data_type 确定哪些按钮可用
        # if data_type not in ["eeg", "fnirs", "ecg"]:  # 假设这些类型支持曲线图
        #     curve_action.setEnabled(False)
        #
        # if data_type not in ["eeg", "fnirs"]:  # 假设这些类型支持柱状图
        #     bar_action.setEnabled(False)
        #
        # if data_type != "eeg":  # 假设只有 EEG 类型支持地形图
        #     topo_action.setEnabled(False)
        #
        # if data_type not in ["fnirs", "eeg"]:  # 假设这些类型支持热力图
        #     heatmap_action.setEnabled(False)
        #
        # if data_type != "network":  # 假设只有 network 类型支持网络图
        #     network_action.setEnabled(False)

        # 连接信号到对应的槽函数
        curve_action.triggered.connect(lambda: self.plot_feature_curve(data, name+'_curve'))
        bar_action.triggered.connect(lambda: self.plot_bar(data, name+'_bar'))
        topo_action.triggered.connect(lambda: self.plot_topomap(data, name+'_topo'))
        heatmap_action.triggered.connect(lambda: self.set_plot_type(item, "热力图"))
        network_action.triggered.connect(lambda: self.set_plot_type(item, "网络图"))
        table_action.triggered.connect(lambda: self.plot_table(data, name+'_table'))

        # 将操作添加到菜单
        menu.addAction(curve_action)
        menu.addAction(bar_action)
        menu.addAction(topo_action)
        # menu.addAction(heatmap_action)
        # menu.addAction(network_action)
        menu.addAction(table_action)

        # 显示菜单
        menu.exec_(self.listWidget.viewport().mapToGlobal(position))

    def set_plot_type(self, item, plot_type):
        """
        设置列表项的绘图类型。
        """
        item.setData(Qt.UserRole, plot_type)
        item.setText(f"{item.text()} - {plot_type}")

    def plot_bar(self, data, name):
        bar_widget = BarPlotWidget(data, data['ch_names'], data['feature'].keys())
        self.drawing_widget.addTab(bar_widget, name)
        self.drawing_widget.setCurrentWidget(bar_widget)

    def plot_feature_curve(self, data, name):
        curve_widget = CurvePlotWidget(data, data['ch_names'], data['feature'].keys())
        self.drawing_widget.addTab(curve_widget, name)
        self.drawing_widget.setCurrentWidget(curve_widget)

    def plot_topomap(self, data, name):
        topomap_widget = TopoMapPlotWidget(data, data['ch_names'], data['feature'].keys())
        self.drawing_widget.addTab(topomap_widget, name)
        self.drawing_widget.setCurrentWidget(topomap_widget)

    def plot_table(self, data, name):
        table_widget = TablePlotWidget(data, data['ch_names'], data['feature'].keys())
        self.drawing_widget.addTab(table_widget, name)
        self.drawing_widget.setCurrentWidget(table_widget)

    def plot_raw_by_file(self, path=None):
        def trans_data(data):
            if data['type'] == 'fnirs_preprocessed':
                result = data['data'][0]
                result.extend(data['data'][1])
                channel = data['ch_names'][0]
                channel.extend(data['ch_names'][1])
                data['data'] = result
                data['ch_names'] = channel
            return data

        data, file_path = read_file_by_qt(self, path)
        print(data)
        if data:
            data = trans_data(data)
            # plot_raw(data=data['data'], channel=data['ch_names'])
            self.drawing_widget = RawCurvePlotDialog(data=data, filePath=file_path[0], parent=self)
            self.drawing_widget.plot_data(self.drawing_widget.current_page)

    def open_folder(self):
        # 打开文件夹选择对话框
        dir_path = QFileDialog.getExistingDirectory(self, "选择文件夹", "")
        if dir_path:
            self.list_files(dir_path)
            self.file_name_lineedit.setText(dir_path)

    def list_files(self, dir_path):
        # 列出目录下所有符合条件的文件
        self.listWidget.clear()  # 清空现有列表项
        for filename in os.listdir(dir_path):
            if filename.endswith(('.edf', '.bdf', '.mat', '.nirs', '.ecg', '.xlsx', '.json')):
                # 创建列表项，仅显示文件名
                item = QListWidgetItem(filename)
                item.setData(32, os.path.join(dir_path, filename))  # 使用自定义角色存储完整路径
                self.listWidget.addItem(item)

    def list_item_double_clicked(self, index):
        # 双击列表项事件
        item = self.listWidget.item(index.row())
        file_path = item.data(32)  # 获取保存在自定义角色的完整路径
        name = item.text()

        # 检查是否已经存在具有相同名称的Tab页面
        for i in range(self.drawing_widget.count()):
            if self.drawing_widget.tabText(i) == name:
                # 如果Tab页面已存在，跳转到该页面
                self.drawing_widget.setCurrentIndex(i)
                return

        plot_widget = self.plot_by_file_type(path=[file_path])
        if plot_widget:
            self.drawing_widget.addTab(plot_widget, name)
            self.drawing_widget.setCurrentWidget(plot_widget)
        # self.clearLayout()

    def clearLayout(self):
        # 清除布局中的所有控件
        while self.drawing_layout.count():
            item = self.drawing_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def plot_by_file_type(self, path):
        def trans_data(data):
            if data['type'] == 'fnirs_preprocessed':
                result = data['data'][0]
                result.extend(data['data'][1])
                channel = data['ch_names'][0]
                channel.extend(data['ch_names'][1])
                data['data'] = result
                data['ch_names'] = channel
            return data

        drawing_widget = None
        if 'subject_01_MI_statistic_result' in path[0]:
            drawing_widget = TestBoxPlot()
        elif 'subject_01_MI_ml_result' in path[0]:
            drawing_widget = TestMLPlot()
        elif 'subject_01_MI_eeg_raw' in path[0]:
            drawing_widget = TestEEGPlot()
        # elif 'subject_01_MI_fnirs' in path[0]:
        #     drawing_widget = TestfNIRSSensorPlot()
        # elif 'subject_01_MI_combine_eeg_fnirs' in path[0]:
        #     drawing_widget = TestEEGandfNIRSSensorPlot()
        else:
            data, file_path = read_file_by_qt(self, path)

            if data:
                if data['type'] in self.curve_type:
                    # plot_raw(data=data['data'], channel=data['ch_names'])
                    if data['type'] == 'fnirs_preprocessed' or data['type'] == 'fnirs':
                        data = trans_data(data)
                        drawing_widget = fNIRSPlotDialog(data=data, filePath=file_path[0], parent=self)
                    else:
                        drawing_widget = RawCurvePlotDialog(data=data, filePath=file_path[0], parent=self)
                    drawing_widget.plot_data(drawing_widget.current_page)

                elif data['type'] in self.topomap_type:
                    drawing_widget = TopomapViewer(data=data, parent=self)
                elif data['type'] in self.time_frequency_type:
                    drawing_widget = TimeFrequencyViewer(data=data, parent=self)

        return drawing_widget

    def close_tab(self, index):
        # 关闭Tab页面
        self.tab_widget.removeTab(index)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = BrainFusionViewer()
    main_window.show()
    sys.exit(app.exec_())
