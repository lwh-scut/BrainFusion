# -*- coding: utf-8 -*-
# @Time    : 2024/5/15 8:37
# @Author  : XXX
# @Site    : 
# @File    : main.py
# @Software: PyCharm 
# @Comment :
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QSizePolicy
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

from BrainFusion.pipeLine.coupling_analysis_with_dialog import NeurovascularCouplingDialog
from BrainFusion.pipeLine.task_with_dialog import TaskDesigner
from BrainFusion.statistic.statistical_analysis_dialog import StatisticalAnalysisDialog
from BrainFusion.utils.BIDS import BIDSConverter
from BrainFusion.viewer.viewer_dialog import BrainFusionViewer
from UI.function_menu import PreprocessMenu, FeatureMenu, NewFeatureMenu, UtilsMenu


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # 初始化UI
        self.initUI()

    def initUI(self):
        # 创建一个垂直布局
        vbox = QVBoxLayout()

        # 添加图片
        self.label = QLabel(self)
        pixmap = QPixmap('resources/logo/brain_fusion(1).png')  # 确保替换为正确的图片路径
        if pixmap.isNull():
            print("Failed to load image!")
        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)  # 让图片适应label大小
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 设置label自动扩展
        self.label.setAlignment(Qt.AlignCenter)  # 内容居中
        vbox.addWidget(self.label)
        vbox.addStretch(1)

        # 添加五个按钮
        self.bnt_task = QPushButton("BrainFusion Task Designer 1.0")
        self.bnt_preprocess = QPushButton("Signal Preprocess")

        self.bnt_feature = QPushButton("Feature Calculation")
        self.bnt_coupling = QPushButton("Coupling Analysis")
        self.bnt_fusion = QPushButton("Fusion Analysis")
        self.bnt_static = QPushButton("Statistical Analysis")
        self.bnt_viewer = QPushButton("BrainFusion Viewer")
        self.bnt_util = QPushButton("Utilities")

        self.bnt_task.clicked.connect(self.open_task_designer)
        self.bnt_preprocess.clicked.connect(self.open_preprocess)
        self.bnt_feature.clicked.connect(self.open_feature)
        self.bnt_viewer.clicked.connect(self.open_viewer)
        self.bnt_coupling.clicked.connect(self.open_coupling)
        self.bnt_static.clicked.connect(self.open_stastic)
        self.bnt_util.clicked.connect(self.open_util)

        # vbox.addWidget(self.bnt_task)
        vbox.addWidget(self.bnt_preprocess)
        vbox.addWidget(self.bnt_feature)
        vbox.addWidget(self.bnt_static)
        vbox.addWidget(self.bnt_coupling)
        # vbox.addWidget(self.bnt_fusion)
        vbox.addWidget(self.bnt_viewer)
        vbox.addWidget(self.bnt_util)
        vbox.addStretch(1)
        # 设置布局
        self.setLayout(vbox)

        # 窗口设置
        self.center_on_screen()
        self.setWindowTitle('brain fusion')
        self.show()

    def open_task_designer(self):
        self.task_designer = TaskDesigner()
        self.task_designer.showMaximized()

    def open_preprocess(self):
        self.preprocess_dialog = PreprocessMenu()
        self.preprocess_dialog.show()

    def open_feature(self):
        self.feature_dialog = NewFeatureMenu()
        self.feature_dialog.show()

    def open_coupling(self):
        self.coupling = NeurovascularCouplingDialog()
        self.coupling.show()

    def open_stastic(self):
        self.stastic = StatisticalAnalysisDialog()
        self.stastic.show()

    def open_fusion(self):
        pass

    def open_viewer(self):
        self.bf_viewer = BrainFusionViewer()
        self.bf_viewer.showMaximized()

    def open_util(self):
        self.utils_menu = UtilsMenu()
        self.utils_menu.show()

    def center_on_screen(self):
        screen_geometry = QApplication.desktop().screenGeometry()
        screen_center = screen_geometry.center()
        self_geometry = self.geometry()
        self_geometry.moveCenter(screen_center)
        self.setGeometry(self_geometry.x(), self_geometry.y(), 300, 400)

def main():
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
