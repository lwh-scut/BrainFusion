import os
import sys

from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import QWidget, QPushButton, QLineEdit, QVBoxLayout, QTreeWidget, QTreeWidgetItem, QHBoxLayout, \
    QFileDialog, QHeaderView, QApplication, QMainWindow, QStackedWidget, QGroupBox, QButtonGroup, QRadioButton, \
    QScrollArea, QLabel, QTextEdit, QTabWidget, QDialog, QDialogButtonBox, QListWidgetItem, QListWidget, QSizePolicy
from PyQt5.QtCore import QDir, Qt, QDateTime, QFileInfo, QEvent


class BFNavigation(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Navigation")
        self.setGeometry(100, 100, 800, 600)
        main_layout = QHBoxLayout()
        main_layout.setSpacing(0)  # Ensure no space between navigation bar and pages
        nav_widget = QWidget()
        nav_widget.setFixedWidth(150)
        self.nav_layout = QVBoxLayout()
        self.nav_layout.setSpacing(0)
        self.nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_widget.setLayout(self.nav_layout)
        nav_widget.setStyleSheet("""
            background-color: #0288D1;
            border-right: 3px solid #34495e;
            color: #ffffff;
            box-shadow: 2px 0 5px rgba(0, 0, 0, 0.3);
        """)  # Navigation bar with shadow for 3D effect

        # Add navigation buttons
        self.buttons = []
        # Right page display area
        self.pages = QStackedWidget()

        main_layout.addWidget(nav_widget)
        main_layout.addWidget(self.pages)

        # Set the main layout to the central widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def add_button_and_page(self, btn_text, new_page):
        button = QPushButton(btn_text)
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # button.setFixedHeight(60)
        button.setStyleSheet("""
            QPushButton {
                background-color: #0288D1;
                color: #ffffff;
                padding: 15px;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton:checked  {
                background-color: #29B6F6;
            }
        """)
        button.setCheckable(True)
        button.clicked.connect(lambda checked, page=new_page: self.switch_page_with_page(page))

        self.buttons.append(button)
        self.nav_layout.addWidget(button)
        self.pages.addWidget(new_page)
        self.buttons[0].setChecked(True)

    def switch_page_with_page(self, page):
        index = self.pages.indexOf(page)
        self.pages.setCurrentIndex(index)
        for i, button in enumerate(self.buttons):
            button.setChecked(self.pages.widget(i) == page)


class BFTabNavigation(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Navigation")
        self.setGeometry(100, 100, 800, 600)

        # Create a QTabWidget with tabs on the top
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 0;
            }
            QTabWidget::tab-bar {
                alignment: center;
            }
            QTabBar::tab {
                background-color: #0969A2;
                color: #ffffff;
                padding: 5px;
                font-size: 16px;
                font-weight: bold;
                border: 1px solid #34495e;
                border-radius: 0; /* No rounded corners */
                margin: 0; /* No margin between tabs */
                
            }
            QTabBar::tab:selected {
                background-color: #0288D1;
                border-bottom: 4px solid #0277BD; /* Stronger contrast for selected tab */
            }
            QTabBar::tab:hover {
                background-color: #0277BD;
            }
            QTabBar::tab:!selected {
                margin-bottom: 2px; /* Small gap for visual separation */
            }
            QTabBar::tab:!selected:hover {
                background-color: #0277BD;
            }
        """)

        # Set the tab widget as the central widget
        self.setCentralWidget(self.tab_widget)

        # Ensure tabs fill the available space
        self.tab_widget.tabBar().setStyleSheet("""
            QTabBar {
                qproperty-drawBase: 0;
            }
            QTabBar::tab {
                width: 200px; /* Equal width for all tabs */
            }
        """)

    def add_tab(self, tab_name, new_page):
        self.tab_widget.addTab(new_page, tab_name)


class BFLogWidget(QWidget):
    def __init__(self):
        super().__init__()

        # Main layout for the log widget
        self.layout = QVBoxLayout()
        self.label = QLabel('Output Information')
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-weight: bold; font-size: 10pt;")

        # QTextEdit for displaying logs
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 10px;
                font-family: Courier;
                font-size: 10pt;
                color: #333;
            }
        """)

        # Buttons for saving and clearing logs
        button_layout = QHBoxLayout()
        self.save_button = BFPushButton("Save Log")
        self.save_button.clicked.connect(self.save_log)
        self.clear_button = BFPushButton("Clear Log")
        self.clear_button.clicked.connect(self.clear_log)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.clear_button)

        # Add the QLabel, QTextEdit, and buttons to the layout
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.log_display)
        self.layout.addLayout(button_layout)

        # Set the layout for this widget
        self.setLayout(self.layout)

    def add_log(self, event, is_error=False, is_success=False, is_start=False, is_end=False):
        current_time = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")

        if is_error:
            log_entry = f"""
            <p style='margin-bottom: 15px; color: red;'>
                <strong>{current_time}</strong><br>
                <strong>[ERROR]</strong> {event}
            </p>
            """
        elif is_success:
            log_entry = f"""
            <p style='margin-bottom: 15px; color: green;'>
                <strong>{current_time}</strong><br>
                <strong>[SUCCESS]</strong> {event}
            </p>
            """
        elif is_start:
            log_entry = f"""
            <p style='margin-bottom: 15px; color: blue;'>
                <strong>{current_time}</strong><br>
                <strong>[START]</strong> {event}
            </p>
            """
        elif is_end:
            log_entry = f"""
            <p style='margin-bottom: 15px; color: blue;'>
                <strong>{current_time}</strong><br>
                <strong>[FINISH]</strong> {event}
            </p>
            """
        else:
            log_entry = f"""
            <p style='margin-bottom: 15px;'>
                <strong>{current_time}</strong><br>
                {event}
            </p>
            """

        self.log_display.append(log_entry)

    def save_log(self):
        log_folder = "log"
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        current_time = QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss")
        log_filename = os.path.join(log_folder, f"log_{current_time}.txt")

        with open(log_filename, "w", encoding='utf-8') as log_file:
            log_file.write(self.log_display.toPlainText())

    def clear_log(self):
        self.log_display.clear()


class BFPanel(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.setStyleSheet("""
            QGroupBox {
                 /* background-color: #ffffff;设置背景色为白色 */
                /* color: #0288D1;  设置标题颜色为蓝色 */
                font-weight: bold; /* 设置标题字体加粗 */
            }
        """)
        self.vlayout = QVBoxLayout(self)
        # self.vlayout.setContentsMargins(0, 0, 0, 0)


class BFGroupBox(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.setStyleSheet("""
            QGroupBox {
                 /* background-color: #ffffff;设置背景色为白色 */
                /* color: #0288D1;  设置标题颜色为蓝色 */
                font-weight: bold; /* 设置标题字体加粗 */
            }
        """)


class BFPushButton(QPushButton):
    def __init__(self, text):
        super().__init__(text)
        self.setStyleSheet(self.button_style())

    def button_style(self):
        return """
        QPushButton {
            background-color: #227fc7;
            color: white;
            border-radius: 6px;
            padding: 6px;
            font-size: 16px;
            font-family: Arial, sans-serif;
        }
        QPushButton:hover {
            background-color: #2293c7;
        }
        QPushButton:pressed {
            background-color: #2272c7;
        }
        """


from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTreeWidget, QTreeWidgetItem, QLabel, QHBoxLayout


class ProcessDialog(QDialog):
    def __init__(self, import_file_list, parent=None):
        super().__init__(parent)
        self.import_file_list = import_file_list
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Processing Files")
        self.resize(800, 300)

        layout = QVBoxLayout()

        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["File", "Processed"])
        self.tree_widget.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        layout.addWidget(self.tree_widget)

        self.add_tree_items()

        self.setLayout(layout)

    def setTitle(self, text):
        self.setWindowTitle(text)

    def add_tree_items(self):
        print(self.import_file_list)
        if len(self.import_file_list) == 1:
            file_path = self.import_file_list[0]
            if not isinstance(file_path, str) and len(file_path) > 1:
                file_path = file_path[0]
            item = QTreeWidgetItem([file_path, "No"])
            item.setTextAlignment(0, Qt.AlignCenter)
            item.setTextAlignment(1, Qt.AlignCenter)
            self.tree_widget.addTopLevelItem(item)
        else:
            for file_path in self.import_file_list:
                if not isinstance(file_path, str) and len(file_path) > 1:
                    file_path = file_path[0]
                item = QTreeWidgetItem([file_path, "No"])
                item.setTextAlignment(0, Qt.AlignCenter)
                item.setTextAlignment(1, Qt.AlignCenter)
                self.tree_widget.addTopLevelItem(item)
        self.tree_widget.header().setDefaultAlignment(Qt.AlignCenter)
        # Set text alignment to center
        self.tree_widget.header().setDefaultAlignment(Qt.AlignCenter)

        # Set styles for "No" and "Yes"
        for i in range(self.tree_widget.topLevelItemCount()):
            item = self.tree_widget.topLevelItem(i)
            if item.text(1) == "No":
                item.setForeground(1, QColor("red"))
                item.setFont(1, QFont("Arial", weight=QFont.Bold))
            elif item.text(1) == "Yes":
                item.setForeground(1, QColor("green"))
                item.setFont(1, QFont("Arial", weight=QFont.Bold))


    def update_status(self, index, status):
        if len(self.import_file_list) == 1:
            item = self.tree_widget.topLevelItem(index)
        else:
            item = self.tree_widget.topLevelItem(index)
        item.setText(1, status)
        if status == "No":
            item.setForeground(1, QColor("red"))
            item.setFont(1, QFont("Arial", weight=QFont.Bold))
        elif status == "Yes":
            item.setForeground(1, QColor("green"))
            item.setFont(1, QFont("Arial", weight=QFont.Bold))


class BFShowMessageEvent(QEvent):
    def __init__(self, message):
        super().__init__(QEvent.User)
        self.message = message


class BFFileListPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.file_list = []  # 初始化文件列表
        self.file_status = {}  # 保存文件的处理状态等信息
        self.show_files = {'processed': True, 'unprocessed': True}  # 初始化显示条件
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        # 文件夹选择部分
        folder_layout = QHBoxLayout()
        self.folder_button = BFPushButton("Select Folder")
        self.folder_button.clicked.connect(self.selectFolder)
        self.folder_lineedit = QLineEdit()
        self.folder_lineedit.setReadOnly(True)
        folder_layout.addWidget(self.folder_button)
        folder_layout.addWidget(self.folder_lineedit)

        layout.addLayout(folder_layout)

        # 文件夹内容显示部分
        self.tree_widget = QTreeWidget()
        self.tree_widget.setColumnCount(4)  # 设置四列
        self.tree_widget.setHeaderLabels(["File Name", "File Type", "Processed", "File Path"])

        # 设置列宽自动调整
        self.tree_widget.header().setSectionResizeMode(QHeaderView.ResizeToContents)

        # 设置列内容居中对齐
        for i in range(4):
            self.tree_widget.headerItem().setTextAlignment(i, Qt.AlignCenter)

        layout.addWidget(self.tree_widget)

        self.tree_widget.itemClicked.connect(self.printFilePath)

        self.setLayout(layout)

    def selectFolder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.folder_lineedit.setText(folder)
            self.populateTree(folder)

    def populateTree(self, folder):
        self.tree_widget.clear()
        self.file_list = []  # 清空文件列表
        self.file_status = {}  # 清空文件状态
        self.populateTreeRecursively(folder)
        self.showFiles()

    def populateTreeRecursively(self, folder):
        dir_iterator = QDir(folder)
        dir_iterator.setFilter(QDir.AllDirs | QDir.NoDotAndDotDot | QDir.Files)

        for entry in dir_iterator.entryInfoList():
            file_info = QFileInfo(entry.filePath())
            if file_info.isDir():
                self.populateTreeRecursively(file_info.absoluteFilePath())
            else:
                file_path = file_info.absoluteFilePath()
                file_name = file_info.fileName()
                file_suffix = file_info.suffix()
                processed = "No"  # 默认未处理
                self.file_list.append(file_path)
                self.file_status[file_path] = {
                    "file_name": file_name,
                    "file_suffix": file_suffix,
                    "processed": processed
                }

    def showFiles(self):
        self.tree_widget.clear()
        for file_path in self.file_list:
            status = self.file_status[file_path]
            if (status["processed"] == "Yes" and self.show_files['processed']) or (
                    status["processed"] == "No" and self.show_files['unprocessed']):
                item = QTreeWidgetItem([status["file_name"], status["file_suffix"], status["processed"], file_path])
                for i in range(4):
                    item.setTextAlignment(i, Qt.AlignCenter)
                self.tree_widget.addTopLevelItem(item)

    def printFilePath(self, item, column):
        if column == 0:  # 点击文件名列时打印路径
            file_path = self.getFilePath(item)
            print("File Path:", file_path)

    def getFilePath(self, item):
        return item.text(3)

    def get_files_by_extension(self, extensions):
        matched_files = [file for file in self.file_list if file.split('.')[-1] in extensions]
        return matched_files

    def updateShowFiles(self, processed=None, unprocessed=None):
        if processed is not None:
            self.show_files['processed'] = processed
        if unprocessed is not None:
            self.show_files['unprocessed'] = unprocessed
        self.showFiles()

    def toggleProcessed(self, file_path):
        if file_path in self.file_status:
            current_status = self.file_status[file_path]["processed"]
            new_status = "Yes" if current_status == "No" else "No"
            self.file_status[file_path]["processed"] = new_status
            self.showFiles()

    def showSpecifiedFiles(self, file_paths):
        self.tree_widget.clear()
        for file_path in file_paths:
            if file_path in self.file_status:
                status = self.file_status[file_path]
                item = QTreeWidgetItem([status["file_name"], status["file_suffix"], status["processed"], file_path])
                for i in range(4):
                    item.setTextAlignment(i, Qt.AlignCenter)
                self.tree_widget.addTopLevelItem(item)


class BFParameterSwitchPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Parameter Switching App")
        self.setGeometry(100, 100, 800, 600)
        # Main layout
        main_layout = QVBoxLayout(self)
        # Upper part: Radio button layout
        self.radio_layout = QHBoxLayout()
        # Button group to manage radio buttons
        self.button_group = QButtonGroup()
        self.button_group.buttonClicked.connect(self.radio_button_clicked)
        # Lower part: Parameter pages
        self.parameter_pages = QStackedWidget()
        self.parameter_pages.setStyleSheet("""
            QScrollArea {
                background-color: #ffffff;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
                box-shadow: inset 0px 0px 5px rgba(0, 0, 0, 0.2);
            }
        """)
        main_layout.addLayout(self.radio_layout)
        main_layout.addWidget(self.parameter_pages)
        main_layout.addStretch(1)
        # Dictionary to store radio buttons and their corresponding parameter pages
        self.radio_dict = {}

    def add_radio_button_with_parameters(self, radio_text, parameter_widget, default=False):
        radio_button = QRadioButton(radio_text)
        self.radio_layout.addWidget(radio_button)
        # Create a scroll area for the parameter widget
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(parameter_widget)

        self.parameter_pages.addWidget(scroll_area)

        self.button_group.addButton(radio_button)

        # Add to dictionary
        self.radio_dict[radio_button] = scroll_area

        # Set default radio button
        if default:
            radio_button.setChecked(True)
            index = self.parameter_pages.indexOf(scroll_area)
            self.parameter_pages.setCurrentIndex(index)

    def radio_button_clicked(self, button):
        index = self.parameter_pages.indexOf(self.radio_dict[button])
        self.parameter_pages.setCurrentIndex(index)


class BFSelectDialog(QDialog):
    def __init__(self, items, parent=None):
        super(BFSelectDialog, self).__init__(parent)
        self.setWindowTitle("Select Items")
        self.layout = QVBoxLayout(self)

        self.listWidget = QListWidget(self)
        for item in items:
            listItem = QListWidgetItem(item)
            listItem.setFlags(listItem.flags() | Qt.ItemIsUserCheckable)
            listItem.setCheckState(Qt.Unchecked)
            self.listWidget.addItem(listItem)

        self.layout.addWidget(self.listWidget)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout.addWidget(self.buttonBox)

    def getSelectedItems(self):
        selectedItems = []
        for index in range(self.listWidget.count()):
            item = self.listWidget.item(index)
            if item.checkState() == Qt.Checked:
                selectedItems.append(item.text())
        return selectedItems


class BFSelectWidget(QWidget):
    def __init__(self, text):
        super(BFSelectWidget, self).__init__()
        self.layout = QHBoxLayout(self)

        self.button = BFPushButton(text)
        self.button.clicked.connect(self.openDialog)
        self.layout.addWidget(self.button)

        self.lineEdit = QLineEdit(self)
        self.lineEdit.setReadOnly(True)
        self.layout.addWidget(self.lineEdit)

        self.items = []

    def openDialog(self):
        dialog = BFSelectDialog(self.items, self)
        if dialog.exec_() == QDialog.Accepted:
            selectedItems = dialog.getSelectedItems()
            self.lineEdit.setText(", ".join(selectedItems))

    def addItems(self, items):
        self.items = items

    def getSelectItem(self):
        return self.lineEdit.text().split(", ")


class BFScrollArea(QScrollArea):
    def __init__(self):
        super(BFScrollArea, self).__init__()
        # 创建一个 QWidget 作为 ScrollArea 的内容
        self.container_widget = QWidget()
        # 创建一个 ScrollArea 并设置它的内容
        self.setWidget(self.container_widget)
        self.setWidgetResizable(True)
        self.setStyleSheet("QScrollArea { border: none; }")

    def set_layout(self, layout):
        self.container_widget.setLayout(layout)


# 使用示例
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    log_widget = BFFileListPanel()
    log_widget.show()
    sys.exit(app.exec_())
