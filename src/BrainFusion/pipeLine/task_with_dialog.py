# -*- coding: utf-8 -*-
# @Time    : 2024/4/16 14:24
# @Author  : XXX
# @Site    : 
# @File    : task_with_dialog.py
# @Software: PyCharm 
# @Comment :
import json
import os
import sys
import threading

import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTreeWidget, QTreeWidgetItem,
                             QTableWidget, QTableWidgetItem, QHeaderView, QVBoxLayout, QHBoxLayout,
                             QMenu, QProgressBar, QCheckBox, QPushButton, QSplitter, QDialog,
                             QMessageBox, QLabel, QVBoxLayout, QFileDialog, QAction, QInputDialog, QGroupBox, QLineEdit,
                             QToolBar, QScrollArea, QDoubleSpinBox, QFormLayout, QComboBox, QSpinBox)
from PyQt5.QtCore import Qt, pyqtSignal

from BrainFusion.io.File_IO import read_file_by_path
from BrainFusion.pipeLine.pipeLine import root_mean_square, variance, mean_absolute_value, zero_crossing, \
    eeg_power_spectral_density, hjorth_parameters, aperiodic_parameters, sample_entropy, eeg_multiscale_entropy, \
    wavelet_transform, wavelet_packet_energy, short_time_Fourier_transform, continuous_wavelet_transform, \
    local_network_features, global_network_features, eeg_microstate
from BrainFusion.pipeLine.pipeLine_with_dialog import ChannelsDialog
from BrainFusion.pipeLine.preprocessing import eeg_preprocessing


class TaskDesigner(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.task_step_list = []

    def initUI(self):
        self.setWindowTitle('Brain Fusion Task Designer')
        self.setGeometry(100, 100, 900, 600)

        # 主布局和窗口中心部件
        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)
        layout = QHBoxLayout(centralWidget)

        # 算法选择树
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel('Algorithm Steps')

        # 树节点结构，根据您之前的代码完整复制
        preprocess_menu = QTreeWidgetItem(self.tree, ['Preprocess'])
        signal_preprocess_menu = QTreeWidgetItem(preprocess_menu, ['Signal Preprocess'])
        signal_preprocess_menu.addChild(QTreeWidgetItem(['EEG']))
        signal_preprocess_menu.addChild(QTreeWidgetItem(['EMG']))
        signal_preprocess_menu.addChild(QTreeWidgetItem(['fNIRS']))
        signal_preprocess_menu.addChild(QTreeWidgetItem(['ECG']))
        signal_preprocess_menu.addChild(QTreeWidgetItem(['Other']))
        preprocess_menu.addChild(QTreeWidgetItem(['Create Epoch']))

        pipeline_menu = QTreeWidgetItem(self.tree, ['Signal Features'])
        time_domain_menu = QTreeWidgetItem(pipeline_menu, ['Time Domain'])
        time_domain_menu.addChild(QTreeWidgetItem(['root mean square']))
        time_domain_menu.addChild(QTreeWidgetItem(['variance']))
        time_domain_menu.addChild(QTreeWidgetItem(['mean absolute value']))
        time_domain_menu.addChild(QTreeWidgetItem(['zero crossing']))
        time_domain_menu.addChild(QTreeWidgetItem(['hjorth']))

        frequency_domain_menu = QTreeWidgetItem(pipeline_menu, ['Frequency Domain'])
        frequency_domain_menu.addChild(QTreeWidgetItem(['power spectral density (eeg)']))
        frequency_domain_menu.addChild(QTreeWidgetItem(['power spectral density']))
        frequency_domain_menu.addChild(QTreeWidgetItem(['aperiodic parameters (eeg)']))

        nonlinear_domain_menu = QTreeWidgetItem(pipeline_menu, ['Nonlinear Domain'])
        nonlinear_domain_menu.addChild(QTreeWidgetItem(['sample entropy']))
        nonlinear_domain_menu.addChild(QTreeWidgetItem(['multiscale entropy']))

        time_frequency_menu = QTreeWidgetItem(pipeline_menu, ['Time-Frequency'])
        time_frequency_menu.addChild(QTreeWidgetItem(['short-time Fourier transform']))
        time_frequency_menu.addChild(QTreeWidgetItem(['wavelet transform']))
        time_frequency_menu.addChild(QTreeWidgetItem(['continuous wavelet transform']))
        time_frequency_menu.addChild(QTreeWidgetItem(['wavelet packet energy']))

        network_menu = QTreeWidgetItem(pipeline_menu, ['Network Analysis'])
        network_menu.addChild(QTreeWidgetItem(['local network']))
        network_menu.addChild(QTreeWidgetItem(['global network']))
        network_menu.addChild(QTreeWidgetItem(['dynamic network']))

        microstate_menu = QTreeWidgetItem(pipeline_menu, ['Microstate'])
        microstate_menu.addChild(QTreeWidgetItem(['microstate (eeg)']))

        multi_signal_menu = QTreeWidgetItem(self.tree, ['Multi-Signal Analysis'])
        coupling_menu = QTreeWidgetItem(multi_signal_menu, ['Signal Coupling'])
        coupling_menu.addChild(QTreeWidgetItem(['neurovascular coupling(NVC)']))
        coupling_menu.addChild(QTreeWidgetItem(['cortical-muscular coupling(CMC)']))
        coupling_menu.addChild(QTreeWidgetItem(['brain-cardiac coupling (BCC)']))

        # 设置右键菜单
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.onRightClickTree)
        self.tree.expandAll()

        # 工具栏
        self.setupToolbar()

        # 任务列表表格
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ['Task Step', 'Progress', 'Parameters', 'Input Data', 'Output Data', 'Save Data Option', 'Save Data Folder',
             'Step Start'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.onRightClickTaskTable)
        self.creator = DatasetCreator()

        # 分隔线
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.tree)
        splitter.addWidget(self.table)
        splitter.addWidget(self.creator)
        splitter.setSizes([300, 1200, 300])

        layout.addWidget(splitter)

        # self.showMaximized()

    def setupToolbar(self):
        toolbar = QToolBar("Toolbar", self)
        self.addToolBar(toolbar)

        btnAddTask = QAction('Create New Task', self)
        btnSaveTask = QAction('Save Task', self)
        btnLoadTask = QAction('Import Task', self)

        toolbar.addAction(btnAddTask)
        toolbar.addAction(btnSaveTask)
        toolbar.addAction(btnLoadTask)

        btnAddTask.triggered.connect(self.addTask)
        # btnSaveTask.triggered.connect(self.saveTaskToFile)
        # btnLoadTask.triggered.connect(self.loadTaskFromFile)

    def addTask(self, task_name):
        row_position = self.table.rowCount()

        self.table.insertRow(row_position)
        self.table.setItem(row_position, 0, QTableWidgetItem(task_name))
        if task_name == 'EEG':
            task_step = EEGPreprocessTaskStep()
        elif task_name == 'root mean square':
            task_step = RootMeanSquareTaskStep()
        elif task_name == 'variance':
            task_step = VarianceTaskStep()
        elif task_name == 'mean absolute value':
            task_step = MeanAbsoluteValueTaskStep()
        elif task_name == 'zero crossing':
            task_step = ZeroCrossingTaskStep()
        elif task_name == 'hjorth':
            task_step = HjorthTaskStep()
        elif task_name == 'power spectral density (eeg)':
            task_step = EEGPowerSpectralDensityTaskStep()
        elif task_name == 'aperiodic parameters (eeg)':
            task_step = AperiodicTaskStep()
        elif task_name == 'sample entropy':
            task_step = SampleEntropyTaskStep()
        elif task_name == 'multiscale entropy':
            task_step = MultiscaleEntropyTaskStep()
        elif task_name == 'short-time Fourier transform':
            task_step = ShortTimeFourierTransformTaskStep()
        elif task_name == 'wavelet transform':
            task_step = WaveletTransformTaskStep()
        elif task_name == 'continuous wavelet transform':
            task_step = ContinuousWaveletTransformTaskStep()
        elif task_name == 'wavelet packet energy':
            task_step = WaveletPacketEnergyTaskStep()
        elif task_name == 'local network':
            task_step = LocalNetworkTaskStep()
        elif task_name == 'global network':
            task_step = GlobalNetworkTaskStep()
        elif task_name == 'microstate (eeg)':
            task_step = EEGMicrostateTaskStep()
        else:
            task_step = TaskStepWidegt()
        if task_step.output_name:
            self.table.setCellWidget(row_position, 4, QLabel(task_step.output_name))
        task_step.getDataGroups.connect(self.getDataGroups)
        task_step.getDataGroupsFiles.connect(self.getDataGroupsFiles)
        task_step.finished.connect(self.creator.add_data_groups_by_dict)
        self.task_step_list.append(task_step)

        self.table.setCellWidget(row_position, 1, task_step.progressBarWidget)
        self.table.setCellWidget(row_position, 2, task_step.param_cellwidget)
        self.table.setCellWidget(row_position, 3, task_step.input_cellwidget)
        self.table.setCellWidget(row_position, 5, task_step.saveDataCheckbox)
        self.table.setCellWidget(row_position, 6, task_step.saveDataCellWidget)
        self.table.setCellWidget(row_position, 7, task_step.start_cellwidget)
        self.table.setRowHeight(row_position, 80)

    def deleteTask(self):
        current_row = self.table.currentRow()
        if current_row != -1:
            self.table.removeRow(current_row)

    def onRightClickTree(self, position):
        menu = QMenu()
        addAction = menu.addAction('Add to Task List')
        action = menu.exec_(self.tree.viewport().mapToGlobal(position))
        if action == addAction:
            item = self.tree.currentItem()
            if item:
                self.addTask(item.text(0))

    def onRightClickTaskTable(self, position):
        menu = QMenu()
        deleteAction = menu.addAction('Delete')
        action = menu.exec_(self.table.viewport().mapToGlobal(position))
        if action == deleteAction:
            self.deleteTask()

    def open_parameters_dialog(self, row):
        dlg = QDialog(self)
        dlg.setWindowTitle("Set Parameters")
        dlg.setLayout(QVBoxLayout())
        dlg.layout().addWidget(QLabel("Adjust parameters here"))
        dlg.setGeometry(300, 300, 200, 100)
        dlg.exec_()

    def openDialog(self, line_edit, data_groups):
        dialog = CustomDialog(data_groups, self)

        if dialog.exec():
            selected_items = dialog.get_selected_items()
            line_edit.setText(", ".join(selected_items))

    def getDataGroups(self):
        # 获取并打印第一层目录的项目
        # top_level_count = self.creator.groupTreeWidget.topLevelItemCount()
        # top_level_item_list = []
        # for i in range(top_level_count):
        #     top_level_item = self.creator.groupTreeWidget.topLevelItem(i).text(0)
        #     top_level_item_list.append(top_level_item)
        self.creator.get_all_paths_by_groups()
        for task in self.task_step_list:
            task.data_groups = []
            for key in FileManager.file_results.keys():
                if FileManager.file_results[key]:
                    task.data_groups.append(key)

    def getDataGroupsFiles(self, id, data_groups):
        raw_paths = {}
        new_paths = {}
        for data_group in data_groups:
            self.creator.get_all_paths_by_groups()
            if FileManager.file_results[data_group]:
                raw_paths[data_group] = FileManager.file_results[data_group].copy()
                new_paths[data_group] = FileManager.save_file_results[data_group].copy()
        for task in self.task_step_list:
            if task.ID == id:
                task.selected_data_groups = raw_paths.copy()
                task.save_path = new_paths.copy()

    def makeSavePath(self, parent_path):
        # 遍历树形结构中的项
        for i in range(self.creator.treeWidget.topLevelItemCount()):
            item = self.creator.treeWidget.topLevelItem(i)
            item_text = item.text(0)  # 获取项的文本内容
            item_path = os.path.join(parent_path, item_text)  # 构建项的完整路径

            # 如果项是文件夹，则创建文件夹并递归处理其子项
            if item.childCount() != 0:
                self.makeSavePath(item_path)


class TaskStepWidegt(QWidget):
    getDataGroups = pyqtSignal(str)
    getDataGroupsFiles = pyqtSignal(int, list)
    finished = pyqtSignal(str, str)
    ID_count = 0

    def __init__(self):
        super().__init__()
        self.ID_count += 1
        self.ID = self.ID_count
        self.step_name = None
        self.data_groups = []
        self.selected_data_groups = None
        self.save_path = None
        self.is_save = False
        self.saveFolder = None
        self.num_files = 0
        self.num_processed_files = 0
        self.output_name = ''
        self.initUI()

    def initUI(self):
        self.progressBar = QProgressBar()
        self.progressBar.setValue(0)
        self.progressBar.setFixedHeight(20)
        self.progressBarWidget = QWidget()
        self.progressBarLayout = QVBoxLayout()
        self.progressBarLayout.setContentsMargins(0, 0, 0, 0)
        self.progressBarLayout.addWidget(self.progressBar)
        self.progressBarLayout.setAlignment(self.progressBar, Qt.AlignCenter)
        self.progressBarWidget.setLayout(self.progressBarLayout)

        self.param_layout = QVBoxLayout()
        self.param_layout.setContentsMargins(0, 2, 0, 2)
        self.param_layout.setSpacing(0)
        self.param_button = QPushButton('Set Parameters')
        self.param_button.setFixedHeight(40)
        self.param_button.clicked.connect(lambda: self.open_parameters_dialog())
        self.param_layout.addWidget(self.param_button)
        self.param_cellwidget = QWidget()
        self.param_cellwidget.setLayout(self.param_layout)

        self.input_layout = QVBoxLayout()
        self.input_layout.setContentsMargins(0, 2, 0, 2)
        self.input_layout.setSpacing(0)
        self.input_lineedit = QLineEdit()
        self.input_lineedit.setPlaceholderText("Selected groups will be displayed here")
        self.input_button = QPushButton("Select")
        self.input_button.clicked.connect(self.openDialog)
        self.input_layout.addWidget(self.input_lineedit)
        self.input_layout.addWidget(self.input_button)
        self.input_cellwidget = QWidget()
        self.input_cellwidget.setLayout(self.input_layout)

        self.saveDataCheckbox = QCheckBox("Save Data?")
        self.saveDataCheckbox.setChecked(True)

        self.saveDataLayout = QVBoxLayout()
        self.saveDataLayout.setSpacing(0)
        self.saveDataLayout.setContentsMargins(0, 2, 0, 2)
        self.saveDataLineEdit = QLineEdit()
        self.saveDataLineEdit.setPlaceholderText("Folder path will be displayed here")
        self.saveDataLayout.addWidget(self.saveDataLineEdit)
        self.chooseFolderButton = QPushButton("Select")
        self.chooseFolderButton.clicked.connect(lambda: self.chooseFolder())
        self.saveDataLayout.addWidget(self.chooseFolderButton)
        self.saveDataCellWidget = QWidget()
        self.saveDataCellWidget.setLayout(self.saveDataLayout)

        self.start_layout = QVBoxLayout()
        self.start_layout.setContentsMargins(0, 2, 0, 2)
        self.start_layout.setSpacing(0)
        self.startButton = QPushButton('Start')
        self.startButton.setFixedHeight(40)
        self.startButton.clicked.connect(lambda: self.startTask())
        self.start_layout.addWidget(self.startButton)
        self.start_cellwidget = QWidget()
        self.start_cellwidget.setLayout(self.start_layout)

    def open_parameters_dialog(self):
        pass

    def prepare(self):
        if self.saveDataCheckbox.isChecked():
            self.is_save = True
        if self.save_path:
            for key in self.save_path.keys():
                paths = self.save_path[key]
                for i in range(len(paths)):
                    self.num_files += 1
                    if len(paths[i]) > 1:
                        self.save_path[key][i] = [self.save_path[key][i][0]]
                    if self.saveFolder:
                        self.save_path[key][i] = [
                            os.path.join(self.saveFolder, self.output_name, self.save_path[key][i][0])]
                        self.save_path[key][i][0] = self.save_path[key][i][0].split('.')[0] + '.mat'
                    if not os.path.exists(os.path.dirname(self.save_path[key][i][0])):
                        os.makedirs(os.path.dirname(self.save_path[key][i][0]))
        self.startButton.setText('Processing...')
        FileManager.file_results[self.output_name] = []
        self.progressBar.setValue(0)
        print(self.selected_data_groups.items())

    def end(self):
        self.startButton.setText('Start')
        self.num_processed_files = 0
        self.num_files = 0
        self.finished.emit(os.path.join(self.saveFolder, self.output_name), self.output_name)

    def chooseFolder(self):
        self.saveFolder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if self.saveFolder:
            self.saveDataLineEdit.setText(self.saveFolder)

    def openDialog(self):
        self.getDataGroups.emit('ok')
        dialog = CustomDialog(self.data_groups)
        dialog.show()
        if dialog.exec():
            selected_items = dialog.get_selected_items()
            self.input_lineedit.setText(", ".join(selected_items))
            self.getDataGroupsFiles.emit(self.ID, selected_items)

    def run(self):
        pass

    def startTask(self):

        # QMessageBox.information(self, "Task Started", f"Task '{str(self.step_name)}' has been started.")
        thread = threading.Thread(target=self.run)
        thread.start()


class EEGPreprocessTaskStep(TaskStepWidegt):
    instance_count = 0

    def __init__(self):
        super().__init__()
        type(self).instance_count += 1
        self.step_name = 'EEG Preprocessing'
        self.parameter = EEGPreprocessParameter()
        self.output_name = self.step_name + '_' + str(type(self).instance_count)

    def __del__(self):
        # 每次实例被销毁时，减少计数
        type(self).instance_count -= 1

    @classmethod
    def get_instance_count(cls):
        # 类方法，返回当前实例数量
        return cls.instance_count

    def open_parameters_dialog(self):
        self.parameter.show()

    def run(self):
        self.prepare()
        parameter_dict = self.parameter.getParameter()
        for data_groups, filePaths in self.selected_data_groups.items():
            for i, filePath in enumerate(filePaths):
                data = read_file_by_path(filePath[0])
                eeg_preprocessing(data=data['data'], chan_list=data['ch_names'], fs=parameter_dict['fs'],
                                  events=data['events'], bad_channels=parameter_dict['bad_channels'],
                                  lowcut=parameter_dict['lowcut'],
                                  highcut=parameter_dict['highcut'],
                                  montage=data['montage'],
                                  filter_order=parameter_dict['filter_order'],
                                  filter=parameter_dict['filter_type'], north_f=parameter_dict['north_f'],
                                  Q=parameter_dict['Q'],
                                  rm_distortion=parameter_dict['rm_distortion'],
                                  rm_persent=parameter_dict['rm_persent'], rm_outlier=parameter_dict['rm_outlier'],
                                  is_ICA=parameter_dict['is_ica_enabled'],
                                  ICA_component=parameter_dict['ica_components'],
                                  ICA_method=parameter_dict['ica_method'],
                                  is_ref=True, ref_chan=parameter_dict['ref_chan_list'], is_save=self.is_save,
                                  save_path=self.save_path[data_groups][i][0],
                                  save_filestyle=parameter_dict['file_type'], is_baseline=parameter_dict['is_baseline'],
                                  baseline_range=parameter_dict['baseline_range'])
                FileManager.file_results[self.output_name].append(self.save_path[data_groups][i][0])
                self.num_processed_files += 1
                self.progressBar.setValue(int(self.num_processed_files / self.num_files * 100))
        self.end()


class EEGPreprocessParameter(QScrollArea):
    def __init__(self):
        super().__init__()
        self.central_widget = QWidget(self)
        self.setWidget(self.central_widget)
        self.setWidgetResizable(True)  # 让内部 widget 可以自动调整大小
        self.layout = QVBoxLayout(self.central_widget)
        self.left_layout = QFormLayout()
        self.layout.addLayout(self.left_layout)

        self.fs_spinbox = QDoubleSpinBox()
        self.fs_spinbox.setRange(1, 10000)
        self.fs_spinbox.setValue(1000)
        self.left_layout.addRow('采样率：', self.fs_spinbox)

        # 添加"剔除坏导"按钮
        btn_remove_bad_channels = QPushButton('剔除坏导')
        btn_remove_bad_channels.clicked.connect(self.show_remove_bad_channels_dialog)
        self.bad_channels_lineedit = QLineEdit(self)
        self.bad_channels_lineedit.setReadOnly(True)
        self.left_layout.addRow(btn_remove_bad_channels, self.bad_channels_lineedit)

        # 创建带通滤波的GroupBox
        filter_groupbox = QGroupBox('带通滤波')
        filter_layout = QFormLayout(filter_groupbox)

        self.lowcut_spinbox = QDoubleSpinBox()
        self.lowcut_spinbox.setRange(0, 10000)
        self.lowcut_spinbox.setValue(2)
        filter_layout.addRow('低通截止频率：', self.lowcut_spinbox)

        self.highcut_spinbox = QDoubleSpinBox()
        self.highcut_spinbox.setRange(0, 10000)
        self.highcut_spinbox.setValue(40)
        filter_layout.addRow('高通截止频率：', self.highcut_spinbox)

        self.filter_combobox = QComboBox()
        self.filter_combobox.addItems(['Butterworth', 'Bessel', 'Chebyshev'])
        filter_layout.addRow('滤波器类型：', self.filter_combobox)

        self.filterorder_spinbox = QSpinBox()
        self.filterorder_spinbox.setRange(0, 20)
        self.filterorder_spinbox.setValue(4)
        filter_layout.addRow('滤波器阶数：', self.filterorder_spinbox)

        self.left_layout.addRow(filter_groupbox)

        # 创建带通滤波的GroupBox
        nortch_groupbox = QGroupBox('工频陷波')
        nortch_layout = QFormLayout(nortch_groupbox)

        self.nortch_spinbox = QDoubleSpinBox()
        self.nortch_spinbox.setRange(0, 10000)
        self.nortch_spinbox.setValue(50.0)
        nortch_layout.addRow('陷波频率：', self.nortch_spinbox)

        self.Q_spinbox = QSpinBox()
        self.Q_spinbox.setRange(0, 50)
        self.Q_spinbox.setValue(30)
        nortch_layout.addRow('品质因数(Q)：', self.Q_spinbox)

        self.left_layout.addRow(nortch_groupbox)

        # 创建去噪的GroupBox
        denoise_groupbox = QGroupBox('去噪相关参数')
        denoise_layout = QFormLayout(denoise_groupbox)

        self.rm_distortion_checkbox = QCheckBox('去除失真段')
        self.rm_distortion_checkbox.setChecked(True)
        denoise_layout.addRow(self.rm_distortion_checkbox)

        self.rm_persent_spinbox = QDoubleSpinBox()
        self.rm_persent_spinbox.setRange(0, 1.0)
        self.rm_persent_spinbox.setValue(0.05)
        self.rm_persent_label = QLabel('失真段比例')
        denoise_layout.addRow(self.rm_persent_label, self.rm_persent_spinbox)

        self.rm_outlier_checkbox = QCheckBox('去除异常点')
        self.rm_outlier_checkbox.setChecked(False)
        self.rm_outlier_checkbox.setVisible(False)
        denoise_layout.addRow(self.rm_outlier_checkbox)

        self.left_layout.addRow(denoise_groupbox)

        # 创建ICA的GroupBox
        ica_groupbox = QGroupBox('ICA相关参数')
        ica_layout = QFormLayout(ica_groupbox)

        self.is_ica_checkbox = QCheckBox('启用ICA')
        self.is_ica_checkbox.setChecked(True)
        ica_layout.addRow(self.is_ica_checkbox)

        self.ica_method_label = QLabel('ICA方法：')
        ica_layout.addRow(self.ica_method_label)

        self.ica_method_combobox = QComboBox()
        self.ica_method_combobox.addItems(['fastica', 'infomax', 'picard'])
        ica_layout.addRow(self.ica_method_label, self.ica_method_combobox)

        self.ica_components_spinbox = QSpinBox()
        self.ica_components_spinbox.setRange(0, 1000)
        ica_layout.addRow('ICA Components', self.ica_components_spinbox)

        self.left_layout.addRow(ica_groupbox)

        # 创建重参考的GroupBox
        ref_groupbox = QGroupBox('重参考相关参数')
        ref_layout = QFormLayout(ref_groupbox)
        btn_ref_chan = QPushButton('选择参考电极')
        btn_ref_chan.clicked.connect(self.show_ref_channels_dialog)
        self.ref_chan_lineedit = QLineEdit()
        ref_layout.addRow(btn_ref_chan, self.ref_chan_lineedit)

        self.ref_method_combobox = QComboBox()
        self.ref_method_combobox.addItems(['平均参考', '其他方法'])
        ref_layout.addRow('重参考方法：', self.ref_method_combobox)

        self.left_layout.addRow(ref_groupbox)

        # 创建基线校正的GroupBox
        baseline_groupbox = QGroupBox('基线校正相关参数')
        baseline_layout = QFormLayout(baseline_groupbox)
        self.is_baseline_checkbox = QCheckBox('启用基线校正')
        self.is_baseline_checkbox.setChecked(False)
        self.baseline_start_spinbox = QSpinBox(self)
        self.baseline_end_spinbox = QSpinBox(self)
        baseline_layout.addRow(self.is_baseline_checkbox)
        baseline_layout.addRow('基线开始时间:', self.baseline_start_spinbox)
        baseline_layout.addRow('基线结束时间:', self.baseline_end_spinbox)

        self.left_layout.addRow(baseline_groupbox)

        # 为 QCheckBox 设置信号槽函数
        self.is_ica_checkbox.stateChanged.connect(self.update_ica_controls_visibility)
        self.rm_distortion_checkbox.stateChanged.connect(self.update_rm_distortion_controls_visibility)

        # 初始状态下设置为不可见
        self.ica_method_label.setVisible(True)
        self.ica_method_combobox.setVisible(True)
        self.rm_persent_label.setVisible(True)
        self.rm_persent_spinbox.setVisible(True)

        self.setGeometry(300, 300, 400, 600)
        self.setWindowTitle('EEG Preprocessing Configuration')

    def update_ica_controls_visibility(self):
        # 根据 QCheckBox 的状态设置 ICA 控件是否可见
        is_ica_enabled = self.is_ica_checkbox.isChecked()
        self.ica_method_label.setVisible(is_ica_enabled)
        self.ica_method_combobox.setVisible(is_ica_enabled)

    def update_rm_distortion_controls_visibility(self):
        # 根据 QCheckBox 的状态设置失真段比例控件是否可见
        is_rm_distortion_enabled = self.rm_distortion_checkbox.isChecked()
        self.rm_persent_spinbox.setVisible(is_rm_distortion_enabled)
        self.rm_persent_label.setVisible(is_rm_distortion_enabled)

    def show_remove_bad_channels_dialog(self):
        if self.data is not None:
            dialog = ChannelsDialog(self.data['ch_names'], self)
            result = dialog.exec_()

            if result == QDialog.Accepted:
                selected_channels = dialog.get_selected_channels()
                result_text = ', '.join(selected_channels)
                self.bad_channels_lineedit.setText(f'{result_text}')
                self.ica_components_spinbox.setValue(self.data['nchan'] - len(selected_channels))
        else:
            QMessageBox.warning(None, 'Warning', '请先导入数据', QMessageBox.Ok)

    def show_ref_channels_dialog(self):
        if self.data is not None:
            dialog = ChannelsDialog(self.data['ch_names'], self)
            result = dialog.exec_()

            if result == QDialog.Accepted:
                selected_channels = dialog.get_selected_channels()
                result_text = ', '.join(selected_channels)
                self.ref_chan_lineedit.setText(f'{result_text}')
        else:
            QMessageBox.warning(None, 'Warning', '请先导入数据', QMessageBox.Ok)

    def getParameter(self):
        parameters = {
            "bad_channels": [chan.replace(' ', '') for chan in self.bad_channels_lineedit.text().split(',')],
            "fs": self.fs_spinbox.value(),
            "lowcut": self.lowcut_spinbox.value(),
            "highcut": self.highcut_spinbox.value(),
            "filter_type": self.filter_combobox.currentText(),
            "filter_order": self.filterorder_spinbox.value(),
            "north_f": self.nortch_spinbox.value(),
            "Q": self.Q_spinbox.value(),
            "is_ica_enabled": self.is_ica_checkbox.isChecked(),
            "ica_method": self.ica_method_combobox.currentText() if self.is_ica_checkbox.isChecked() else None,
            "ica_components": self.ica_components_spinbox.value(),
            "rm_distortion": self.rm_distortion_checkbox.isChecked(),
            "rm_persent": self.rm_persent_spinbox.value(),
            "rm_outlier": self.rm_outlier_checkbox.isChecked(),
            "ref_chan_list": [chan.replace(' ', '') for chan in self.ref_chan_lineedit.text().split(',')],
            "ref_method": self.ref_method_combobox.currentText(),
            "is_baseline": self.is_baseline_checkbox.isChecked(),
            "baseline_range": (self.baseline_start_spinbox.value(), self.baseline_end_spinbox.value()),
            "output_file": '',
            "file_type": 'mat'
        }
        return parameters


class RootMeanSquareTaskStep(TaskStepWidegt):
    instance_count = 0

    def __init__(self):
        super().__init__()
        type(self).instance_count += 1
        self.step_name = 'rms'
        self.parameter = RootMeanSquareParameter()
        self.output_name = self.step_name + '_' + str(type(self).instance_count)

    def __del__(self):
        # 每次实例被销毁时，减少计数
        type(self).instance_count -= 1

    @classmethod
    def get_instance_count(cls):
        # 类方法，返回当前实例数量
        return cls.instance_count

    def open_parameters_dialog(self):
        self.parameter.show()

    def run(self):
        self.prepare()
        parameter_dict = self.parameter.getParameter()
        for data_groups, filePaths in self.selected_data_groups.items():
            for i, filePath in enumerate(filePaths):
                data = read_file_by_path(filePath[0])
                root_mean_square(data=data, is_save=self.is_save, is_sliding=parameter_dict['is_sliding'],
                                 window_size=parameter_dict['window_size'],
                                 overlap_rate=parameter_dict['overlap_rate'],
                                 save_path=self.save_path[data_groups][i][0],
                                 save_filestyle=parameter_dict['file_type'])
                FileManager.file_results[self.output_name].append(self.save_path[data_groups][i][0])
                self.num_processed_files += 1
                self.progressBar.setValue(int(self.num_processed_files / self.num_files * 100))
        self.end()


class VarianceTaskStep(TaskStepWidegt):
    instance_count = 0

    def __init__(self):
        super().__init__()
        type(self).instance_count += 1
        self.step_name = 'var'
        self.parameter = VarianceParameter()
        self.output_name = self.step_name + '_' + str(type(self).instance_count)

    def __del__(self):
        # 每次实例被销毁时，减少计数
        type(self).instance_count -= 1

    @classmethod
    def get_instance_count(cls):
        # 类方法，返回当前实例数量
        return cls.instance_count

    def open_parameters_dialog(self):
        self.parameter.show()

    def run(self):
        self.prepare()
        parameter_dict = self.parameter.getParameter()
        for data_groups, filePaths in self.selected_data_groups.items():
            for i, filePath in enumerate(filePaths):
                data = read_file_by_path(filePath[0])
                variance(data=data, is_save=self.is_save, is_sliding=parameter_dict['is_sliding'],
                         window_size=parameter_dict['window_size'],
                         overlap_rate=parameter_dict['overlap_rate'],
                         save_path=self.save_path[data_groups][i][0],
                         save_filestyle=parameter_dict['file_type'])
                FileManager.file_results[self.output_name].append(self.save_path[data_groups][i][0])
                self.num_processed_files += 1
                self.progressBar.setValue(int(self.num_processed_files / self.num_files * 100))
        self.end()


class MeanAbsoluteValueTaskStep(TaskStepWidegt):
    instance_count = 0

    def __init__(self):
        super().__init__()
        type(self).instance_count += 1
        self.step_name = 'mav'
        self.parameter = MeanAbsoluteValueParameter()
        self.output_name = self.step_name + '_' + str(type(self).instance_count)

    def __del__(self):
        # 每次实例被销毁时，减少计数
        type(self).instance_count -= 1

    @classmethod
    def get_instance_count(cls):
        # 类方法，返回当前实例数量
        return cls.instance_count

    def open_parameters_dialog(self):
        self.parameter.show()

    def run(self):
        self.prepare()
        parameter_dict = self.parameter.getParameter()
        for data_groups, filePaths in self.selected_data_groups.items():
            for i, filePath in enumerate(filePaths):
                data = read_file_by_path(filePath[0])
                mean_absolute_value(data=data, is_save=self.is_save, is_sliding=parameter_dict['is_sliding'],
                                    window_size=parameter_dict['window_size'],
                                    overlap_rate=parameter_dict['overlap_rate'],
                                    save_path=self.save_path[data_groups][i][0],
                                    save_filestyle=parameter_dict['file_type'])
                FileManager.file_results[self.output_name].append(self.save_path[data_groups][i][0])
                self.num_processed_files += 1
                self.progressBar.setValue(int(self.num_processed_files / self.num_files * 100))
        self.end()


class ZeroCrossingTaskStep(TaskStepWidegt):
    instance_count = 0

    def __init__(self):
        super().__init__()
        type(self).instance_count += 1
        self.step_name = 'zc'
        self.parameter = ZeroCrossingParameter()
        self.output_name = self.step_name + '_' + str(type(self).instance_count)

    def __del__(self):
        # 每次实例被销毁时，减少计数
        type(self).instance_count -= 1

    @classmethod
    def get_instance_count(cls):
        # 类方法，返回当前实例数量
        return cls.instance_count

    def open_parameters_dialog(self):
        self.parameter.show()

    def run(self):
        self.prepare()
        parameter_dict = self.parameter.getParameter()
        for data_groups, filePaths in self.selected_data_groups.items():
            for i, filePath in enumerate(filePaths):
                data = read_file_by_path(filePath[0])
                zero_crossing(data=data, is_save=self.is_save, is_sliding=parameter_dict['is_sliding'],
                              window_size=parameter_dict['window_size'],
                              overlap_rate=parameter_dict['overlap_rate'],
                              save_path=self.save_path[data_groups][i][0],
                              save_filestyle=parameter_dict['file_type'])
                FileManager.file_results[self.output_name].append(self.save_path[data_groups][i][0])
                self.num_processed_files += 1
                self.progressBar.setValue(int(self.num_processed_files / self.num_files * 100))
        self.end()


class HjorthTaskStep(TaskStepWidegt):
    instance_count = 0

    def __init__(self):
        super().__init__()
        type(self).instance_count += 1
        self.step_name = 'hjorth'
        self.parameter = HjorthParameter()
        self.output_name = self.step_name + '_' + str(type(self).instance_count)

    def __del__(self):
        # 每次实例被销毁时，减少计数
        type(self).instance_count -= 1

    @classmethod
    def get_instance_count(cls):
        # 类方法，返回当前实例数量
        return cls.instance_count

    def open_parameters_dialog(self):
        self.parameter.show()

    def run(self):
        self.prepare()
        parameter_dict = self.parameter.getParameter()
        for data_groups, filePaths in self.selected_data_groups.items():
            for i, filePath in enumerate(filePaths):
                data = read_file_by_path(filePath[0])
                hjorth_parameters(data=data, is_save=self.is_save, is_sliding=parameter_dict['is_sliding'],
                                  window_size=parameter_dict['window_size'],
                                  overlap_rate=parameter_dict['overlap_rate'],
                                  save_path=self.save_path[data_groups][i][0],
                                  save_filestyle=parameter_dict['file_type'])
                FileManager.file_results[self.output_name].append(self.save_path[data_groups][i][0])
                self.num_processed_files += 1
                self.progressBar.setValue(int(self.num_processed_files / self.num_files * 100))
        self.end()


class EEGPowerSpectralDensityTaskStep(TaskStepWidegt):
    instance_count = 0

    def __init__(self):
        super().__init__()
        type(self).instance_count += 1
        self.step_name = 'eeg_psd'
        self.parameter = None
        self.param_button.setVisible(False)
        self.output_name = self.step_name + '_' + str(type(self).instance_count)

    def __del__(self):
        # 每次实例被销毁时，减少计数
        type(self).instance_count -= 1

    @classmethod
    def get_instance_count(cls):
        # 类方法，返回当前实例数量
        return cls.instance_count

    def open_parameters_dialog(self):
        pass

    def run(self):
        self.prepare()
        for data_groups, filePaths in self.selected_data_groups.items():
            for i, filePath in enumerate(filePaths):
                data = read_file_by_path(filePath[0])
                eeg_power_spectral_density(data=data, is_save=self.is_save, save_path=self.save_path[data_groups][i][0],
                                           save_filestyle='mat')
                FileManager.file_results[self.output_name].append(self.save_path[data_groups][i][0])
                self.num_processed_files += 1
                self.progressBar.setValue(int(self.num_processed_files / self.num_files * 100))
        self.end()


class AperiodicTaskStep(TaskStepWidegt):
    instance_count = 0

    def __init__(self):
        super().__init__()
        type(self).instance_count += 1
        self.step_name = 'aperiodic'
        self.parameter = None
        self.param_button.setVisible(False)
        self.output_name = self.step_name + '_' + str(type(self).instance_count)

    def __del__(self):
        # 每次实例被销毁时，减少计数
        type(self).instance_count -= 1

    @classmethod
    def get_instance_count(cls):
        # 类方法，返回当前实例数量
        return cls.instance_count

    def open_parameters_dialog(self):
        pass

    def run(self):
        self.prepare()
        for data_groups, filePaths in self.selected_data_groups.items():
            for i, filePath in enumerate(filePaths):
                data = read_file_by_path(filePath[0])
                aperiodic_parameters(data=data, is_save=self.is_save, save_path=self.save_path[data_groups][i][0],
                                     save_filestyle='mat')
                FileManager.file_results[self.output_name].append(self.save_path[data_groups][i][0])
                self.num_processed_files += 1
                self.progressBar.setValue(int(self.num_processed_files / self.num_files * 100))
        self.end()


class SampleEntropyTaskStep(TaskStepWidegt):
    instance_count = 0

    def __init__(self):
        super().__init__()
        type(self).instance_count += 1
        self.step_name = 'sample entropy'
        self.parameter = None
        self.param_button.setVisible(False)
        self.output_name = self.step_name + '_' + str(type(self).instance_count)

    def __del__(self):
        # 每次实例被销毁时，减少计数
        type(self).instance_count -= 1

    @classmethod
    def get_instance_count(cls):
        # 类方法，返回当前实例数量
        return cls.instance_count

    def open_parameters_dialog(self):
        pass

    def run(self):
        self.prepare()
        for data_groups, filePaths in self.selected_data_groups.items():
            for i, filePath in enumerate(filePaths):
                data = read_file_by_path(filePath[0])
                sample_entropy(data=data, is_save=self.is_save, save_path=self.save_path[data_groups][i][0],
                                     save_filestyle='mat')
                FileManager.file_results[self.output_name].append(self.save_path[data_groups][i][0])
                self.num_processed_files += 1
                self.progressBar.setValue(int(self.num_processed_files / self.num_files * 100))
        self.end()


class MultiscaleEntropyTaskStep(TaskStepWidegt):
    instance_count = 0

    def __init__(self):
        super().__init__()
        type(self).instance_count += 1
        self.step_name = 'multiscale entropy'
        self.parameter = None
        self.param_button.setVisible(False)
        self.output_name = self.step_name + '_' + str(type(self).instance_count)

    def __del__(self):
        # 每次实例被销毁时，减少计数
        type(self).instance_count -= 1

    @classmethod
    def get_instance_count(cls):
        # 类方法，返回当前实例数量
        return cls.instance_count

    def open_parameters_dialog(self):
        pass

    def run(self):
        self.prepare()
        for data_groups, filePaths in self.selected_data_groups.items():
            for i, filePath in enumerate(filePaths):
                data = read_file_by_path(filePath[0])
                eeg_multiscale_entropy(data=data, is_save=self.is_save, save_path=self.save_path[data_groups][i][0],
                                     save_filestyle='mat')
                FileManager.file_results[self.output_name].append(self.save_path[data_groups][i][0])
                self.num_processed_files += 1
                self.progressBar.setValue(int(self.num_processed_files / self.num_files * 100))
        self.end()


class WaveletTransformTaskStep(TaskStepWidegt):
    instance_count = 0

    def __init__(self):
        super().__init__()
        type(self).instance_count += 1
        self.step_name = 'dwt'
        self.parameter = WaveletTransformParameter()
        self.output_name = self.step_name + '_' + str(type(self).instance_count)

    def __del__(self):
        # 每次实例被销毁时，减少计数
        type(self).instance_count -= 1

    @classmethod
    def get_instance_count(cls):
        # 类方法，返回当前实例数量
        return cls.instance_count

    def open_parameters_dialog(self):
        self.parameter.show()

    def run(self):
        self.prepare()
        parameter_dict = self.parameter.getParameter()
        for data_groups, filePaths in self.selected_data_groups.items():
            for i, filePath in enumerate(filePaths):
                data = read_file_by_path(filePath[0])
                wavelet_transform(data=data, level=parameter_dict['level'], basis_function=parameter_dict['basis_function'], is_save=self.is_save,
                                  save_path=self.save_path[data_groups][i][0],
                                  save_filestyle='mat')
                FileManager.file_results[self.output_name].append(self.save_path[data_groups][i][0])
                self.num_processed_files += 1
                self.progressBar.setValue(int(self.num_processed_files / self.num_files * 100))
        self.end()


class WaveletPacketEnergyTaskStep(TaskStepWidegt):
    instance_count = 0

    def __init__(self):
        super().__init__()
        type(self).instance_count += 1
        self.step_name = 'wpe'
        self.parameter = WaveletPacketEnergyParameter()
        self.output_name = self.step_name + '_' + str(type(self).instance_count)

    def __del__(self):
        # 每次实例被销毁时，减少计数
        type(self).instance_count -= 1

    @classmethod
    def get_instance_count(cls):
        # 类方法，返回当前实例数量
        return cls.instance_count

    def open_parameters_dialog(self):
        self.parameter.show()

    def run(self):
        self.prepare()
        parameter_dict = self.parameter.getParameter()
        for data_groups, filePaths in self.selected_data_groups.items():
            for i, filePath in enumerate(filePaths):
                data = read_file_by_path(filePath[0])
                wavelet_packet_energy(data=data, level=parameter_dict['level'], basis_function=parameter_dict['basis_function'], is_save=self.is_save,
                                  save_path=self.save_path[data_groups][i][0],
                                  save_filestyle='mat')
                FileManager.file_results[self.output_name].append(self.save_path[data_groups][i][0])
                self.num_processed_files += 1
                self.progressBar.setValue(int(self.num_processed_files / self.num_files * 100))
        self.end()


class ShortTimeFourierTransformTaskStep(TaskStepWidegt):
    instance_count = 0

    def __init__(self):
        super().__init__()
        type(self).instance_count += 1
        self.step_name = 'stft'
        self.parameter = ShortTimeFourierTransformParameter()
        self.output_name = self.step_name + '_' + str(type(self).instance_count)

    def __del__(self):
        # 每次实例被销毁时，减少计数
        type(self).instance_count -= 1

    @classmethod
    def get_instance_count(cls):
        # 类方法，返回当前实例数量
        return cls.instance_count

    def open_parameters_dialog(self):
        self.parameter.show()

    def run(self):
        self.prepare()
        parameter_dict = self.parameter.getParameter()
        for data_groups, filePaths in self.selected_data_groups.items():
            for i, filePath in enumerate(filePaths):
                data = read_file_by_path(filePath[0])
                short_time_Fourier_transform(data=data, nperseg=parameter_dict['window_size'],
                                             noverlap=parameter_dict['noverlap'],
                                             window_method='hamming', is_save=self.is_save,
                                             save_path=self.save_path[data_groups][i][0],
                                             save_filestyle='mat')
                FileManager.file_results[self.output_name].append(self.save_path[data_groups][i][0])
                self.num_processed_files += 1
                self.progressBar.setValue(int(self.num_processed_files / self.num_files * 100))
        self.end()


class ContinuousWaveletTransformTaskStep(TaskStepWidegt):
    instance_count = 0

    def __init__(self):
        super().__init__()
        type(self).instance_count += 1
        self.step_name = 'cwt'
        self.parameter = ContinuousWaveletTransformParameter()
        self.output_name = self.step_name + '_' + str(type(self).instance_count)

    def __del__(self):
        # 每次实例被销毁时，减少计数
        type(self).instance_count -= 1

    @classmethod
    def get_instance_count(cls):
        # 类方法，返回当前实例数量
        return cls.instance_count

    def open_parameters_dialog(self):
        self.parameter.show()

    def run(self):
        self.prepare()
        parameter_dict = self.parameter.getParameter()
        for data_groups, filePaths in self.selected_data_groups.items():
            for i, filePath in enumerate(filePaths):
                data = read_file_by_path(filePath[0])
                continuous_wavelet_transform(data, widths=parameter_dict['widths'],
                                             basis_function=parameter_dict['basis_function'],
                                             is_save=self.is_save,
                                             save_path=self.save_path[data_groups][i][0],
                                             save_filestyle='mat')
                FileManager.file_results[self.output_name].append(self.save_path[data_groups][i][0])
                self.num_processed_files += 1
                self.progressBar.setValue(int(self.num_processed_files / self.num_files * 100))
        self.end()


class LocalNetworkTaskStep(TaskStepWidegt):
    instance_count = 0

    def __init__(self):
        super().__init__()
        type(self).instance_count += 1
        self.step_name = 'local network'
        self.parameter = LocalNetworkParameter()
        self.output_name = self.step_name + '_' + str(type(self).instance_count)

    def __del__(self):
        # 每次实例被销毁时，减少计数
        type(self).instance_count -= 1

    @classmethod
    def get_instance_count(cls):
        # 类方法，返回当前实例数量
        return cls.instance_count

    def open_parameters_dialog(self):
        self.parameter.show()

    def run(self):
        self.prepare()
        parameter_dict = self.parameter.getParameter()
        for data_groups, filePaths in self.selected_data_groups.items():
            for i, filePath in enumerate(filePaths):
                data = read_file_by_path(filePath[0])
                local_network_features(data, edge_retention_rate=parameter_dict['edge_retention_rate'],
                                       is_relative_thresholds=parameter_dict['is_relative'], threshold=parameter_dict['threshold'],
                                       method=parameter_dict['method'],
                                       is_save=self.is_save, save_path=self.save_path[data_groups][i][0],
                                       save_filestyle='mat')
                FileManager.file_results[self.output_name].append(self.save_path[data_groups][i][0])
                self.num_processed_files += 1
                self.progressBar.setValue(int(self.num_processed_files / self.num_files * 100))
        self.end()


class GlobalNetworkTaskStep(TaskStepWidegt):
    instance_count = 0

    def __init__(self):
        super().__init__()
        type(self).instance_count += 1
        self.step_name = 'global network'
        self.parameter = GlobalNetworkParameter()
        self.output_name = self.step_name + '_' + str(type(self).instance_count)

    def __del__(self):
        # 每次实例被销毁时，减少计数
        type(self).instance_count -= 1

    @classmethod
    def get_instance_count(cls):
        # 类方法，返回当前实例数量
        return cls.instance_count

    def open_parameters_dialog(self):
        self.parameter.show()

    def run(self):
        self.prepare()
        parameter_dict = self.parameter.getParameter()
        for data_groups, filePaths in self.selected_data_groups.items():
            for i, filePath in enumerate(filePaths):
                data = read_file_by_path(filePath[0])
                global_network_features(data, edge_retention_rate=parameter_dict['edge_retention_rate'],
                                       is_relative_thresholds=parameter_dict['is_relative'], threshold=parameter_dict['threshold'],
                                       method=parameter_dict['method'],
                                       is_save=self.is_save, save_path=self.save_path[data_groups][i][0],
                                       save_filestyle='mat')
                FileManager.file_results[self.output_name].append(self.save_path[data_groups][i][0])
                self.num_processed_files += 1
                self.progressBar.setValue(int(self.num_processed_files / self.num_files * 100))
        self.end()


class EEGMicrostateTaskStep(TaskStepWidegt):
    instance_count = 0

    def __init__(self):
        super().__init__()
        type(self).instance_count += 1
        self.step_name = 'eeg_microstate'
        self.parameter = EEGMicrostateParameter()
        self.output_name = self.step_name + '_' + str(type(self).instance_count)

    def __del__(self):
        # 每次实例被销毁时，减少计数
        type(self).instance_count -= 1

    @classmethod
    def get_instance_count(cls):
        # 类方法，返回当前实例数量
        return cls.instance_count

    def open_parameters_dialog(self):
        self.parameter.show()

    def run(self):
        self.prepare()
        parameter_dict = self.parameter.getParameter()
        for data_groups, filePaths in self.selected_data_groups.items():
            for i, filePath in enumerate(filePaths):
                data = read_file_by_path(filePath[0])
                eeg_microstate(data=data, n_clusters=parameter_dict['n_clusters'],
                               peak_threshold=parameter_dict['peak_threshold'], is_show=False,
                               is_save=self.is_save, save_path=self.save_path[data_groups][i][0],
                               save_filestyle='mat')
                FileManager.file_results[self.output_name].append(self.save_path[data_groups][i][0])
                self.num_processed_files += 1
                self.progressBar.setValue(int(self.num_processed_files / self.num_files * 100))
        self.end()


class RootMeanSquareParameter(QScrollArea):
    def __init__(self):
        super().__init__()
        self.central_widget = QWidget(self)
        self.setWidget(self.central_widget)
        self.setWidgetResizable(True)  # 让内部 widget 可以自动调整大小
        self.layout = QVBoxLayout(self.central_widget)
        self.left_layout = QFormLayout()
        self.layout.addLayout(self.left_layout)
        self.setWindowTitle('Root Mean Square Dialog')
        # 创建ICA的GroupBox
        self.rms_groupbox = QGroupBox('RMS相关参数')
        rms_layout = QFormLayout(self.rms_groupbox)

        self.is_sliding_checkbox = QCheckBox('启用滑动窗口')
        self.is_sliding_checkbox.setChecked(True)
        rms_layout.addRow(self.is_sliding_checkbox)

        self.sliding_groupbox = QGroupBox()
        sliding_layout = QFormLayout(self.sliding_groupbox)

        self.window_size_spinbox = QSpinBox()
        self.window_size_spinbox.setRange(0, 1000)
        sliding_layout.addRow('窗口大小', self.window_size_spinbox)

        self.overlap_rate_spinbox = QDoubleSpinBox()
        self.overlap_rate_spinbox.setRange(0, 1)
        sliding_layout.addRow('重叠率', self.overlap_rate_spinbox)
        rms_layout.addRow(self.sliding_groupbox)

        self.left_layout.addRow(self.rms_groupbox)

        # 为 QCheckBox 设置信号槽函数
        self.is_sliding_checkbox.stateChanged.connect(self.update_sliding_controls_visibility)

        # 初始状态下设置为不可见
        self.rms_groupbox.setVisible(True)

    def update_sliding_controls_visibility(self):
        # 根据 QCheckBox 的状态设置 ICA 控件是否可见
        is_sliding_enabled = self.is_sliding_checkbox.isChecked()
        self.sliding_groupbox.setVisible(is_sliding_enabled)

    def getParameter(self):
        parameters = {
            "output_file": '',
            "is_sliding": self.is_sliding_checkbox.isChecked(),
            "window_size": self.window_size_spinbox.value(),
            "overlap_rate": self.overlap_rate_spinbox.value(),
            "file_type": 'mat'
        }
        return parameters


class VarianceParameter(QScrollArea):
    def __init__(self):
        super().__init__()
        self.central_widget = QWidget(self)
        self.setWidget(self.central_widget)
        self.setWidgetResizable(True)  # 让内部 widget 可以自动调整大小
        self.layout = QVBoxLayout(self.central_widget)
        self.left_layout = QFormLayout()
        self.layout.addLayout(self.left_layout)
        self.setWindowTitle('Variance Dialog')
        self.var_groupbox = QGroupBox('var相关参数')
        var_layout = QFormLayout(self.var_groupbox)

        self.is_sliding_checkbox = QCheckBox('启用滑动窗口')
        self.is_sliding_checkbox.setChecked(True)
        var_layout.addRow(self.is_sliding_checkbox)

        self.sliding_groupbox = QGroupBox()
        sliding_layout = QFormLayout(self.sliding_groupbox)

        self.window_size_spinbox = QSpinBox()
        self.window_size_spinbox.setRange(0, 1000)
        sliding_layout.addRow('窗口大小', self.window_size_spinbox)

        self.overlap_rate_spinbox = QDoubleSpinBox()
        self.overlap_rate_spinbox.setRange(0, 1)
        sliding_layout.addRow('重叠率', self.overlap_rate_spinbox)
        var_layout.addRow(self.sliding_groupbox)

        self.left_layout.addRow(self.var_groupbox)

        # 为 QCheckBox 设置信号槽函数
        self.is_sliding_checkbox.stateChanged.connect(self.update_sliding_controls_visibility)

        # 初始状态下设置为不可见
        self.var_groupbox.setVisible(True)

    def update_sliding_controls_visibility(self):
        # 根据 QCheckBox 的状态设置 ICA 控件是否可见
        is_sliding_enabled = self.is_sliding_checkbox.isChecked()
        self.sliding_groupbox.setVisible(is_sliding_enabled)

    def getParameter(self):
        parameters = {
            "output_file": '',
            "is_sliding": self.is_sliding_checkbox.isChecked(),
            "window_size": self.window_size_spinbox.value(),
            "overlap_rate": self.overlap_rate_spinbox.value(),
            "file_type": 'mat'
        }
        return parameters


class MeanAbsoluteValueParameter(QScrollArea):
    def __init__(self):
        super().__init__()
        self.central_widget = QWidget(self)
        self.setWidget(self.central_widget)
        self.setWidgetResizable(True)  # 让内部 widget 可以自动调整大小
        self.layout = QVBoxLayout(self.central_widget)
        self.left_layout = QFormLayout()
        self.layout.addLayout(self.left_layout)
        self.setWindowTitle('Mean Absolute Value Dialog')
        self.mav_groupbox = QGroupBox('mav相关参数')
        mav_layout = QFormLayout(self.mav_groupbox)

        self.is_sliding_checkbox = QCheckBox('启用滑动窗口')
        self.is_sliding_checkbox.setChecked(True)
        mav_layout.addRow(self.is_sliding_checkbox)

        self.sliding_groupbox = QGroupBox()
        sliding_layout = QFormLayout(self.sliding_groupbox)

        self.window_size_spinbox = QSpinBox()
        self.window_size_spinbox.setRange(0, 1000)
        sliding_layout.addRow('窗口大小', self.window_size_spinbox)

        self.overlap_rate_spinbox = QDoubleSpinBox()
        self.overlap_rate_spinbox.setRange(0, 1)
        sliding_layout.addRow('重叠率', self.overlap_rate_spinbox)
        mav_layout.addRow(self.sliding_groupbox)

        self.left_layout.addRow(self.mav_groupbox)

        # 为 QCheckBox 设置信号槽函数
        self.is_sliding_checkbox.stateChanged.connect(self.update_sliding_controls_visibility)

        # 初始状态下设置为不可见
        self.mav_groupbox.setVisible(True)

    def update_sliding_controls_visibility(self):
        # 根据 QCheckBox 的状态设置 ICA 控件是否可见
        is_sliding_enabled = self.is_sliding_checkbox.isChecked()
        self.sliding_groupbox.setVisible(is_sliding_enabled)

    def getParameter(self):
        parameters = {
            "output_file": '',
            "is_sliding": self.is_sliding_checkbox.isChecked(),
            "window_size": self.window_size_spinbox.value(),
            "overlap_rate": self.overlap_rate_spinbox.value(),
            "file_type": 'mat'
        }
        return parameters


class ZeroCrossingParameter(QScrollArea):
    def __init__(self):
        super().__init__()
        self.central_widget = QWidget(self)
        self.setWidget(self.central_widget)
        self.setWidgetResizable(True)  # 让内部 widget 可以自动调整大小
        self.layout = QVBoxLayout(self.central_widget)
        self.left_layout = QFormLayout()
        self.layout.addLayout(self.left_layout)
        self.setWindowTitle('Zero Crossing Dialog')
        self.zc_groupbox = QGroupBox('zc相关参数')
        zc_layout = QFormLayout(self.zc_groupbox)

        self.is_sliding_checkbox = QCheckBox('启用滑动窗口')
        self.is_sliding_checkbox.setChecked(True)
        zc_layout.addRow(self.is_sliding_checkbox)

        self.sliding_groupbox = QGroupBox()
        sliding_layout = QFormLayout(self.sliding_groupbox)

        self.window_size_spinbox = QSpinBox()
        self.window_size_spinbox.setRange(0, 1000)
        sliding_layout.addRow('窗口大小', self.window_size_spinbox)

        self.overlap_rate_spinbox = QDoubleSpinBox()
        self.overlap_rate_spinbox.setRange(0, 1)
        sliding_layout.addRow('重叠率', self.overlap_rate_spinbox)
        zc_layout.addRow(self.sliding_groupbox)

        self.left_layout.addRow(self.zc_groupbox)

        # 为 QCheckBox 设置信号槽函数
        self.is_sliding_checkbox.stateChanged.connect(self.update_sliding_controls_visibility)

        # 初始状态下设置为不可见
        self.zc_groupbox.setVisible(True)

    def update_sliding_controls_visibility(self):
        # 根据 QCheckBox 的状态设置 ICA 控件是否可见
        is_sliding_enabled = self.is_sliding_checkbox.isChecked()
        self.sliding_groupbox.setVisible(is_sliding_enabled)

    def getParameter(self):
        parameters = {
            "output_file": '',
            "is_sliding": self.is_sliding_checkbox.isChecked(),
            "window_size": self.window_size_spinbox.value(),
            "overlap_rate": self.overlap_rate_spinbox.value(),
            "file_type": 'mat'
        }
        return parameters


class HjorthParameter(QScrollArea):
    def __init__(self):
        super().__init__()
        self.central_widget = QWidget(self)
        self.setWidget(self.central_widget)
        self.setWidgetResizable(True)  # 让内部 widget 可以自动调整大小
        self.layout = QVBoxLayout(self.central_widget)
        self.left_layout = QFormLayout()
        self.layout.addLayout(self.left_layout)
        self.setWindowTitle('Hjorth Parameters Dialog')
        self.hjorth_groupbox = QGroupBox('Hjorth相关参数')
        hjorth_layout = QFormLayout(self.hjorth_groupbox)

        self.is_sliding_checkbox = QCheckBox('启用滑动窗口')
        self.is_sliding_checkbox.setChecked(True)
        hjorth_layout.addRow(self.is_sliding_checkbox)

        self.sliding_groupbox = QGroupBox()
        sliding_layout = QFormLayout(self.sliding_groupbox)

        self.window_size_spinbox = QSpinBox()
        self.window_size_spinbox.setRange(0, 1000)
        sliding_layout.addRow('窗口大小', self.window_size_spinbox)

        self.overlap_rate_spinbox = QDoubleSpinBox()
        self.overlap_rate_spinbox.setRange(0, 1)
        sliding_layout.addRow('重叠率', self.overlap_rate_spinbox)
        hjorth_layout.addRow(self.sliding_groupbox)

        self.left_layout.addRow(self.hjorth_groupbox)

        # 为 QCheckBox 设置信号槽函数
        self.is_sliding_checkbox.stateChanged.connect(self.update_sliding_controls_visibility)

        # 初始状态下设置为不可见
        self.sliding_groupbox.setVisible(True)

    def update_sliding_controls_visibility(self):
        # 根据 QCheckBox 的状态设置 ICA 控件是否可见
        is_sliding_enabled = self.is_sliding_checkbox.isChecked()
        self.sliding_groupbox.setVisible(is_sliding_enabled)

    def getParameter(self):
        parameters = {
            "output_file": '',
            "is_sliding": self.is_sliding_checkbox.isChecked(),
            "window_size": self.window_size_spinbox.value(),
            "overlap_rate": self.overlap_rate_spinbox.value(),
            "file_type": 'mat'
        }
        return parameters


class WaveletTransformParameter(QScrollArea):
    def __init__(self):
        super().__init__()
        self.central_widget = QWidget(self)
        self.setWidget(self.central_widget)
        self.setWidgetResizable(True)  # 让内部 widget 可以自动调整大小
        self.layout = QVBoxLayout(self.central_widget)
        self.left_layout = QFormLayout()
        self.layout.addLayout(self.left_layout)
        self.setWindowTitle('Wavelet Transform Dialog')
        self.dwt_groupbox = QGroupBox('dwt相关参数')
        dwt_layout = QFormLayout(self.dwt_groupbox)

        self.level_spinbox = QSpinBox()
        self.level_spinbox.setRange(0, 1000)
        dwt_layout.addRow('level', self.level_spinbox)

        self.basis_function_combox = QComboBox()
        self.basis_function_combox.addItems(['db1', 'db2', 'db3', 'db4'])
        dwt_layout.addRow('小波基函数', self.basis_function_combox)

        self.left_layout.addRow(self.dwt_groupbox)

    def getParameter(self):
        parameters = {
            "output_file": '',
            "level": self.level_spinbox.value(),
            "basis_function": self.basis_function_combox.currentText(),
            "file_type": 'mat'
        }
        return parameters


class WaveletPacketEnergyParameter(QScrollArea):
    def __init__(self):
        super().__init__()
        self.central_widget = QWidget(self)
        self.setWidget(self.central_widget)
        self.setWidgetResizable(True)  # 让内部 widget 可以自动调整大小
        self.layout = QVBoxLayout(self.central_widget)
        self.left_layout = QFormLayout()
        self.layout.addLayout(self.left_layout)
        self.setWindowTitle('Wavelet Packet Energy Dialog')
        self.dwt_groupbox = QGroupBox('wpe相关参数')
        dwt_layout = QFormLayout(self.dwt_groupbox)

        self.level_spinbox = QSpinBox()
        self.level_spinbox.setRange(0, 1000)
        dwt_layout.addRow('level', self.level_spinbox)

        self.basis_function_combox = QComboBox()
        self.basis_function_combox.addItems(['db1', 'db2', 'db3', 'db4'])
        dwt_layout.addRow('小波基函数', self.basis_function_combox)

        self.left_layout.addRow(self.dwt_groupbox)

    def getParameter(self):
        parameters = {
            "output_file": '',
            "level": self.level_spinbox.value(),
            "basis_function": self.basis_function_combox.currentText(),
            "file_type": 'mat'
        }
        return parameters


class ShortTimeFourierTransformParameter(QScrollArea):
    def __init__(self):
        super().__init__()
        self.central_widget = QWidget(self)
        self.setWidget(self.central_widget)
        self.setWidgetResizable(True)  # 让内部 widget 可以自动调整大小
        self.layout = QVBoxLayout(self.central_widget)
        self.left_layout = QFormLayout()
        self.layout.addLayout(self.left_layout)
        self.setWindowTitle('Short Time Fourier Transform Dialog')
        self.stft_groupbox = QGroupBox('stft相关参数')
        stft_layout = QFormLayout(self.stft_groupbox)

        self.window_size_spinbox = QSpinBox()
        self.window_size_spinbox.setRange(0, 100000)
        stft_layout.addRow('时间窗大小', self.window_size_spinbox)

        self.noverlap_spinbox = QDoubleSpinBox()
        self.noverlap_spinbox.setRange(0, 1)
        stft_layout.addRow('重叠率', self.noverlap_spinbox)

        self.left_layout.addRow(self.stft_groupbox)

    def getParameter(self):
        parameters = {
            "output_file": '',
            "window_size": self.window_size_spinbox.value(),
            "noverlap": self.noverlap_spinbox.value(),
            "file_type": 'mat'
        }
        return parameters


class ContinuousWaveletTransformParameter(QScrollArea):
    def __init__(self):
        super().__init__()
        self.central_widget = QWidget(self)
        self.setWidget(self.central_widget)
        self.setWidgetResizable(True)  # 让内部 widget 可以自动调整大小
        self.layout = QVBoxLayout(self.central_widget)
        self.left_layout = QFormLayout()
        self.layout.addLayout(self.left_layout)
        self.setWindowTitle('Continuous Wavelet Transform Dialog')
        self.cwt_groupbox = QGroupBox('cwt相关参数')
        cwt_layout = QFormLayout(self.cwt_groupbox)

        self.low_scale_spinbox = QSpinBox()
        self.low_scale_spinbox.setRange(0, 100000)
        cwt_layout.addRow('尺度范围（低）：', self.low_scale_spinbox)

        self.high_scale_spinbox = QSpinBox()
        self.high_scale_spinbox.setRange(0, 100000)
        cwt_layout.addRow('尺度范围（高）：', self.high_scale_spinbox)

        self.basis_function_combox = QComboBox()
        self.basis_function_combox.addItems(['cmor'])
        cwt_layout.addRow('小波基函数：', self.basis_function_combox)

        self.left_layout.addRow(self.cwt_groupbox)

    def getParameter(self):
        parameters = {
            "output_file": '',
            "widths": np.arange(self.low_scale_spinbox.value(), self.high_scale_spinbox.value()),
            "basis_function": self.basis_function_combox.currentText(),
            "file_type": 'mat'
        }
        return parameters


class LocalNetworkParameter(QScrollArea):
    def __init__(self):
        super().__init__()
        self.central_widget = QWidget(self)
        self.setWidget(self.central_widget)
        self.setWidgetResizable(True)  # 让内部 widget 可以自动调整大小
        self.layout = QVBoxLayout(self.central_widget)
        self.left_layout = QFormLayout()
        self.layout.addLayout(self.left_layout)
        self.setWindowTitle('Local Network Dialog')
        self.local_network_groupbox = QGroupBox('Local Network相关参数')
        local_network_layout = QFormLayout(self.local_network_groupbox)
        self.method_combox = QComboBox()
        self.method_combox.addItems(['cov'])
        self.edge_retention_rate_spinbox = QDoubleSpinBox()
        self.edge_retention_rate_spinbox.setRange(0, 1)
        local_network_layout.addRow('边保留率', self.edge_retention_rate_spinbox)

        self.is_absolute_checkbox = QCheckBox('使用绝对阈值')
        self.is_absolute_checkbox.setChecked(False)
        local_network_layout.addRow(self.is_absolute_checkbox)

        self.absolute_groupbox = QGroupBox()
        absolute_layout = QFormLayout(self.absolute_groupbox)

        self.threshold_spinbox = QSpinBox()
        self.threshold_spinbox.setRange(0, 1000)
        absolute_layout.addRow('绝对阈值', self.threshold_spinbox)
        local_network_layout.addRow(self.absolute_groupbox)

        self.left_layout.addRow(self.local_network_groupbox)

        # 为 QCheckBox 设置信号槽函数
        self.is_absolute_checkbox.stateChanged.connect(self.update_sliding_controls_visibility)

        # 初始状态下设置为不可见
        self.absolute_groupbox.setVisible(False)

    def update_sliding_controls_visibility(self):
        # 根据 QCheckBox 的状态设置 ICA 控件是否可见
        is_sliding_enabled = self.is_absolute_checkbox.isChecked()
        self.is_absolute_checkbox.setVisible(is_sliding_enabled)

    def getParameter(self):
        parameters = {
            "output_file": '',
            "edge_retention_rate": self.edge_retention_rate_spinbox.value(),
            "is_relative": self.is_absolute_checkbox.isChecked(),
            "threshold": self.threshold_spinbox.value(),
            "method": self.method_combox.currentText(),
            "file_type": 'mat'
        }
        return parameters


class GlobalNetworkParameter(QScrollArea):
    def __init__(self):
        super().__init__()
        self.central_widget = QWidget(self)
        self.setWidget(self.central_widget)
        self.setWidgetResizable(True)  # 让内部 widget 可以自动调整大小
        self.layout = QVBoxLayout(self.central_widget)
        self.left_layout = QFormLayout()
        self.layout.addLayout(self.left_layout)
        self.setWindowTitle('Global Network Dialog')
        self.global_network_groupbox = QGroupBox('Global Network相关参数')
        global_network_layout = QFormLayout(self.global_network_groupbox)
        self.method_combox = QComboBox()
        self.method_combox.addItems(['cov'])
        self.edge_retention_rate_spinbox = QDoubleSpinBox()
        self.edge_retention_rate_spinbox.setRange(0, 1)
        global_network_layout.addRow('边保留率', self.edge_retention_rate_spinbox)

        self.is_absolute_checkbox = QCheckBox('使用绝对阈值')
        self.is_absolute_checkbox.setChecked(False)
        global_network_layout.addRow(self.is_absolute_checkbox)

        self.absolute_groupbox = QGroupBox()
        absolute_layout = QFormLayout(self.absolute_groupbox)

        self.threshold_spinbox = QSpinBox()
        self.threshold_spinbox.setRange(0, 1000)
        absolute_layout.addRow('绝对阈值', self.threshold_spinbox)
        global_network_layout.addRow(self.absolute_groupbox)

        self.left_layout.addRow(self.global_network_groupbox)

        # 为 QCheckBox 设置信号槽函数
        self.is_absolute_checkbox.stateChanged.connect(self.update_sliding_controls_visibility)

        # 初始状态下设置为不可见
        self.absolute_groupbox.setVisible(False)

    def update_sliding_controls_visibility(self):
        # 根据 QCheckBox 的状态设置 ICA 控件是否可见
        is_sliding_enabled = self.is_absolute_checkbox.isChecked()
        self.is_absolute_checkbox.setVisible(is_sliding_enabled)

    def getParameter(self):
        parameters = {
            "output_file": '',
            "edge_retention_rate": self.edge_retention_rate_spinbox.value(),
            "is_relative": self.is_absolute_checkbox.isChecked(),
            "threshold": self.threshold_spinbox.value(),
            "method": self.method_combox.currentText(),
            "file_type": 'mat'
        }
        return parameters


class EEGMicrostateParameter(QScrollArea):
    def __init__(self):
        super().__init__()
        self.central_widget = QWidget(self)
        self.setWidget(self.central_widget)
        self.setWidgetResizable(True)  # 让内部 widget 可以自动调整大小
        self.layout = QVBoxLayout(self.central_widget)
        self.left_layout = QFormLayout()
        self.layout.addLayout(self.left_layout)
        self.setWindowTitle('Microstate')
        # 创建mse参数的GroupBox
        microstate_parameter_groupbox = QGroupBox('Microstate计算相关参数')
        microstate_parameter_layout = QFormLayout(microstate_parameter_groupbox)
        self.n_clusters_spinbox = QSpinBox()
        self.n_clusters_spinbox.setRange(1, 10)
        self.n_clusters_spinbox.setValue(4)
        microstate_parameter_layout.addRow('状态数量：', self.n_clusters_spinbox)
        self.peak_threshold_spinbox = QDoubleSpinBox()
        self.peak_threshold_spinbox.setRange(0, 1)
        self.peak_threshold_spinbox.setValue(0)
        microstate_parameter_layout.addRow('GFP峰值阈值：', self.peak_threshold_spinbox)
        self.left_layout.addRow(microstate_parameter_groupbox)

    def getParameter(self):
        parameters = {
            "output_file": '',
            "n_clusters": self.n_clusters_spinbox.value(),
            "peak_threshold": self.peak_threshold_spinbox.value(),
            "file_type": 'mat'
        }
        return parameters


class CustomDialog(QDialog):
    def __init__(self, data_groups, parent=None):
        super().__init__(parent)
        self.data_groups = data_groups
        self.selected_items = []
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        container = QWidget()
        scroll_layout = QVBoxLayout(container)

        for item in self.data_groups:
            cb = QCheckBox(item, container)
            cb.stateChanged.connect(self.update_selection)
            scroll_layout.addWidget(cb)

        scroll_area.setWidget(container)
        layout.addWidget(scroll_area)

        btn_ok = QPushButton("确定", self)
        btn_ok.clicked.connect(self.accept)
        layout.addWidget(btn_ok)

        self.setLayout(layout)
        self.setWindowTitle("选择数据")

    def update_selection(self, state):
        checkbox = self.sender()
        if state == Qt.Checked:
            self.selected_items.append(checkbox.text())
        else:
            self.selected_items.remove(checkbox.text())

    def get_selected_items(self):
        return self.selected_items


class FileManager:
    file_results = {}
    save_file_results = {}


class DatasetCreator(QWidget):
    data_groups = {}

    def __init__(self):
        super().__init__()
        self.title = "Dataset Creator"
        self.taskList = []
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 800, 600)

        mainLayout = QVBoxLayout(self)  # 主布局改为垂直布局

        # 水平布局用于容纳源目录和数据组
        vLayout = QVBoxLayout()
        self.setupGroupArea(vLayout)
        self.setupDatasetArea(vLayout)
        mainLayout.addLayout(vLayout)  # 将水平布局添加到主布局

    def setupDatasetArea(self, layout):
        groupBox = QGroupBox("Source Directory")
        vbox = QVBoxLayout()

        self.btnBrowse = QPushButton('Browsing Folders')
        self.btnBrowse.setFixedWidth(150)
        self.btnBrowse.clicked.connect(lambda: self.open_directory(self.treeWidget))
        vbox.addWidget(self.btnBrowse)

        self.treeWidget = QTreeWidget()
        self.treeWidget.setHeaderLabel("Source Directory")
        self.treeWidget.setSelectionMode(QTreeWidget.ExtendedSelection)
        self.treeWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.treeWidget.customContextMenuRequested.connect(lambda point: self.on_context_menu(self.treeWidget, point))
        vbox.addWidget(self.treeWidget)

        groupBox.setLayout(vbox)
        layout.addWidget(groupBox)

    def setupGroupArea(self, layout):
        groupBox = QGroupBox("Data Groups")
        vbox = QVBoxLayout(groupBox)  # 将groupBox作为布局的父容器

        # 创建一个水平布局来存放按钮
        buttonsLayout = QHBoxLayout()
        self.btnAddGroup = QPushButton('New Data Group')
        self.btnSaveData = QPushButton('Save Data Groups')
        self.btnLoadData = QPushButton('Load Data Groups')
        self.btnAddGroup.setFixedWidth(150)
        self.btnSaveData.setFixedWidth(150)
        self.btnLoadData.setFixedWidth(150)
        self.btnAddGroup.clicked.connect(self.add_new_group)
        self.btnSaveData.clicked.connect(self.save_data_groups)
        self.btnLoadData.clicked.connect(self.load_data_groups)

        buttonsLayout.addWidget(self.btnAddGroup)
        buttonsLayout.addWidget(self.btnSaveData)
        buttonsLayout.addWidget(self.btnLoadData)
        buttonsLayout.addStretch(1)
        vbox.addLayout(buttonsLayout)  # 将按钮的水平布局添加到垂直布局中

        self.groupTreeWidget = QTreeWidget()
        self.groupTreeWidget.setHeaderLabel("Data Groups")
        self.groupTreeWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.groupTreeWidget.customContextMenuRequested.connect(self.on_group_context_menu)
        vbox.addWidget(self.groupTreeWidget)

        layout.addWidget(groupBox)  # 将groupBox添加到传入的layout中

    def open_directory(self, treeWidget):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            treeWidget.clear()
            self.populate_tree_widget(folder, treeWidget.invisibleRootItem())

    def populate_tree_widget(self, path, parent_item):
        entries = sorted(os.listdir(path), key=lambda x: (os.path.isfile(os.path.join(path, x)), x.lower()))
        for entry in entries:
            entry_path = os.path.join(path, entry)
            child_item = QTreeWidgetItem(parent_item, [entry])
            child_item.setData(0, Qt.UserRole, entry_path)
            if os.path.isdir(entry_path):
                self.populate_tree_widget(entry_path, child_item)

    def on_context_menu(self, treeWidget, point):
        items = treeWidget.selectedItems()
        if items:
            menu = QMenu(treeWidget)
            deleteAction = QAction('Delete', treeWidget)
            addToGroupAction = QAction('Add to Group...', treeWidget)
            menu.addAction(deleteAction)
            menu.addAction(addToGroupAction)
            deleteAction.triggered.connect(lambda: self.delete_items(items, treeWidget))
            addToGroupAction.triggered.connect(lambda: self.add_to_group(items))
            menu.exec_(treeWidget.mapToGlobal(point))

    def delete_items(self, items, treeWidget):
        for item in items:
            (item.parent() or treeWidget.invisibleRootItem()).removeChild(item)

    def add_to_group(self, items):
        group_names = [self.groupTreeWidget.topLevelItem(i).text(0) for i in
                       range(self.groupTreeWidget.topLevelItemCount())]
        selected_group, ok = QInputDialog.getItem(self, "Selecting a Data Group", "Selecting a Data Group:", group_names, 0, False)
        if selected_group in FileManager.file_results.keys():
            FileManager.file_results.pop(selected_group)
        if ok and selected_group:
            group_item = next(
                (self.groupTreeWidget.topLevelItem(i) for i in range(self.groupTreeWidget.topLevelItemCount()) if
                 self.groupTreeWidget.topLevelItem(i).text(0) == selected_group), None)
            if group_item:
                self.add_items_to_group_recursive(items, group_item)

    def add_items_to_group_recursive(self, items, target_group_item):
        for item in items:
            # Check for duplicates in the target group item
            duplicate_found = False
            for i in range(target_group_item.childCount()):
                child = target_group_item.child(i)
                if child.text(0) == item.text(0):
                    duplicate_found = True
                    break
            if not duplicate_found:
                new_item = QTreeWidgetItem(target_group_item, [item.text(0)])
                new_item.setData(0, Qt.UserRole, item.data(0, Qt.UserRole))
                # Recursively add children if the item is a directory with children
                if item.childCount() > 0:
                    self.add_items_to_group_recursive([item.child(i) for i in range(item.childCount())], new_item)

    def add_new_group(self):
        group_name, ok = QInputDialog.getText(self, 'Adding a new data group', 'Input data group name:')
        if ok and group_name:
            QTreeWidgetItem(self.groupTreeWidget, [group_name])

    def on_group_context_menu(self, point):
        item = self.groupTreeWidget.itemAt(point)
        if item:
            menu = QMenu(self.groupTreeWidget)
            deleteAction = QAction('Delete', self.groupTreeWidget)
            menu.addAction(deleteAction)
            deleteAction.triggered.connect(lambda: self.delete_items([item], self.groupTreeWidget))
            menu.exec_(self.groupTreeWidget.mapToGlobal(point))

    def save_data_groups(self):
        data_groups = {}
        for i in range(self.groupTreeWidget.topLevelItemCount()):
            group = self.groupTreeWidget.topLevelItem(i)
            group_data = []
            self.collect_data(group, group_data)
            data_groups[group.text(0)] = group_data

        filename, _ = QFileDialog.getSaveFileName(self, "Save Data Groups", "", "JSON Files (*.json)")
        if filename:
            with open(filename, 'w', encoding='utf-8') as file:  # 指定文件以UTF-8编码保存
                json.dump(data_groups, file, ensure_ascii=False, indent=4)  # 使用ensure_ascii=False来处理非ASCII字符

    def collect_data(self, tree_item, data_list):
        for i in range(tree_item.childCount()):
            child = tree_item.child(i)
            child_data = {
                "name": child.text(0),
                "path": child.data(0, Qt.UserRole),
                "children": []
            }
            self.collect_data(child, child_data["children"])
            data_list.append(child_data)

    def get_all_paths_by_groups(self):
        def traverse_tree(item, parent_path, file_paths, save_paths):
            item_path = os.path.join(parent_path, item.text(0))
            if item.childCount() == 0:  # 如果当前项没有子项，说明是叶子节点（文件）
                file_path = item.data(0, Qt.UserRole)
                if file_path and (file_path.endswith("bdf") or file_path.endswith("edf") or file_path.endswith("mat")):
                    file_paths.append(file_path)
                if item.text(0).endswith("bdf") or item.text(0).endswith("edf") or item.text(0).endswith("mat"):
                    save_paths.append(item_path)
            else:
                # Recursively traverse children
                for i in range(item.childCount()):
                    traverse_tree(item.child(i), item_path, file_paths, save_paths)

        def merge_paths(files):
            merged = {}
            for file_path in files:
                folder_path = os.path.dirname(file_path)
                if folder_path in merged:
                    merged[folder_path].append(file_path)
                else:
                    merged[folder_path] = [file_path]
            return merged

        # Traverse top-level items
        for i in range(self.groupTreeWidget.topLevelItemCount()):
            file_paths = []
            save_paths = []
            traverse_tree(self.groupTreeWidget.topLevelItem(i), '', file_paths, save_paths)
            data_group = self.groupTreeWidget.topLevelItem(i).text(0)
            merged_paths = merge_paths(file_paths)
            merged_save_paths = merge_paths(save_paths)
            FileManager.file_results[data_group] = [value for value in merged_paths.values()]
            FileManager.save_file_results[data_group] = [value for value in merged_save_paths.values()]

        print(FileManager.file_results)
        print(FileManager.save_file_results)

    def load_data_groups(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Data Groups", "", "JSON Files (*.json)")
        if filename:
            with open(filename, 'r', encoding='utf-8') as file:
                data_groups = json.load(file)
                self.groupTreeWidget.clear()
                for group_name, items in data_groups.items():
                    group_item = QTreeWidgetItem(self.groupTreeWidget, [group_name])
                    self.recreate_group_structure(group_item, items)

    def check_group_exist(self, group_name):
        for i in range(self.groupTreeWidget.topLevelItemCount()):
            if self.groupTreeWidget.topLevelItem(i).text(0) == group_name:
                group_name += '_2'
                group_name = self.check_group_exist(group_name)
        return group_name

    def add_data_groups_by_dict(self, folder, group_name):
        treeWidget = QTreeWidget()
        self.populate_tree_widget(folder, treeWidget.invisibleRootItem())
        group_name = self.check_group_exist(group_name)
        QTreeWidgetItem(self.groupTreeWidget, [group_name])
        group_item = next(
            (self.groupTreeWidget.topLevelItem(i) for i in range(self.groupTreeWidget.topLevelItemCount()) if
             self.groupTreeWidget.topLevelItem(i).text(0) == group_name), None)
        items = [treeWidget.topLevelItem(i) for i in range(treeWidget.topLevelItemCount())]
        try:
            self.add_items_to_group_recursive(items, group_item)
        except Exception as e:
            print(e)

    def recreate_group_structure(self, parent_item, items):
        for item_data in items:
            item = QTreeWidgetItem(parent_item, [item_data["name"]])
            item.setData(0, Qt.UserRole, item_data["path"])
            if "children" in item_data:
                self.recreate_group_structure(item, item_data["children"])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TaskDesigner()
    sys.exit(app.exec_())
