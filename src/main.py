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
    """Main application window for BrainFusion."""

    def __init__(self):
        """Initialize main application window."""
        super().__init__()
        self.initUI()

    def initUI(self):
        """
        Initialize user interface components.
        """
        # Create vertical layout
        vbox = QVBoxLayout()

        # Add logo image
        self.label = QLabel(self)
        pixmap = QPixmap('resources/logo/brain_fusion(1).png')  # Update with correct path
        if pixmap.isNull():
            print("Failed to load image!")
        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)  # Scale image to fit label
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Enable label expansion
        self.label.setAlignment(Qt.AlignCenter)  # Center content
        vbox.addWidget(self.label)
        vbox.addStretch(1)

        # Create navigation buttons
        self.bnt_task = QPushButton("BrainFusion Task Designer 1.0")
        self.bnt_preprocess = QPushButton("Signal Preprocess")
        self.bnt_feature = QPushButton("Feature Calculation")
        self.bnt_coupling = QPushButton("Coupling Analysis")
        self.bnt_fusion = QPushButton("Fusion Analysis")
        self.bnt_static = QPushButton("Statistical Analysis")
        self.bnt_viewer = QPushButton("BrainFusion Viewer")
        self.bnt_util = QPushButton("Utilities")

        # Connect button signals
        self.bnt_task.clicked.connect(self.open_task_designer)
        self.bnt_preprocess.clicked.connect(self.open_preprocess)
        self.bnt_feature.clicked.connect(self.open_feature)
        self.bnt_viewer.clicked.connect(self.open_viewer)
        self.bnt_coupling.clicked.connect(self.open_coupling)
        self.bnt_static.clicked.connect(self.open_stastic)
        self.bnt_util.clicked.connect(self.open_util)

        # Add buttons to layout
        # vbox.addWidget(self.bnt_task)
        vbox.addWidget(self.bnt_preprocess)
        vbox.addWidget(self.bnt_feature)
        vbox.addWidget(self.bnt_static)
        vbox.addWidget(self.bnt_coupling)
        # vbox.addWidget(self.bnt_fusion)
        vbox.addWidget(self.bnt_viewer)
        vbox.addWidget(self.bnt_util)
        vbox.addStretch(1)

        # Set layout
        self.setLayout(vbox)

        # Configure window
        self.center_on_screen()
        self.setWindowTitle('Brain Fusion')
        self.show()

    def open_task_designer(self):
        """
        Open task designer window.
        """
        self.task_designer = TaskDesigner()
        self.task_designer.showMaximized()

    def open_preprocess(self):
        """
        Open signal preprocessing dialog.
        """
        self.preprocess_dialog = PreprocessMenu()
        self.preprocess_dialog.show()

    def open_feature(self):
        """
        Open feature calculation window.
        """
        self.feature_dialog = NewFeatureMenu()
        self.feature_dialog.show()

    def open_coupling(self):
        """
        Open neurovascular coupling analysis.
        """
        self.coupling = NeurovascularCouplingDialog()
        self.coupling.show()

    def open_stastic(self):
        """
        Open statistical analysis window.
        """
        self.stastic = StatisticalAnalysisDialog()
        self.stastic.show()

    def open_fusion(self):
        """Reserved for future fusion analysis feature."""
        pass

    def open_viewer(self):
        """
        Open data visualization viewer. Launches the BrainFusion Viewer in maximized mode.
        """
        self.bf_viewer = BrainFusionViewer()
        self.bf_viewer.showMaximized()

    def open_util(self):
        """
        Open utilities menu. Displays the utilities menu.
        """
        self.utils_menu = UtilsMenu()
        self.utils_menu.show()

    def center_on_screen(self):
        """
        Center window on screen. Calculates screen dimensions and positions window at center.
        """
        screen_geometry = QApplication.desktop().screenGeometry()
        screen_center = screen_geometry.center()
        self_geometry = self.geometry()
        self_geometry.moveCenter(screen_center)
        self.setGeometry(self_geometry.x(), self_geometry.y(), 300, 400)


def main():
    """
    Main application entry point.

    :return: Application exit code
    :rtype: int
    """
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()