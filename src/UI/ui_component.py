import os
import sys

from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import QWidget, QPushButton, QLineEdit, QVBoxLayout, QTreeWidget, QTreeWidgetItem, QHBoxLayout, \
    QFileDialog, QHeaderView, QApplication, QMainWindow, QStackedWidget, QGroupBox, QButtonGroup, QRadioButton, \
    QScrollArea, QLabel, QTextEdit, QTabWidget, QDialog, QDialogButtonBox, QListWidgetItem, QListWidget, QSizePolicy
from PyQt5.QtCore import QDir, Qt, QDateTime, QFileInfo, QEvent


class BFNavigation(QMainWindow):
    """Main navigation interface with sidebar navigation."""

    def __init__(self):
        """
        Initialize navigation interface.

        Creates a sidebar navigation system with expandable page display.
        """
        super().__init__()
        self.setWindowTitle("Navigation")
        self.setGeometry(100, 100, 800, 600)
        self.init_ui()

    def init_ui(self):
        """Initialize user interface components."""
        # Create main layout
        main_layout = QHBoxLayout()
        main_layout.setSpacing(0)

        # Create navigation sidebar
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
        """)

        # Initialize navigation buttons and pages
        self.buttons = []
        self.pages = QStackedWidget()

        # Configure layout
        main_layout.addWidget(nav_widget)
        main_layout.addWidget(self.pages)

        # Set central widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def add_button_and_page(self, btn_text, new_page):
        """
        Add navigation button and corresponding page.

        :param btn_text: Text to display on navigation button
        :type btn_text: str
        :param new_page: Page widget to display when button is clicked
        :type new_page: QWidget
        """
        # Create navigation button
        button = QPushButton(btn_text)
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
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

        # Add components to interface
        self.buttons.append(button)
        self.nav_layout.addWidget(button)
        self.pages.addWidget(new_page)
        if len(self.buttons) == 1:
            self.buttons[0].setChecked(True)

    def switch_page_with_page(self, page):
        """
        Switch to specified page in display area.

        :param page: Page widget to display
        :type page: QWidget
        """
        index = self.pages.indexOf(page)
        self.pages.setCurrentIndex(index)
        for i, button in enumerate(self.buttons):
            button.setChecked(self.pages.widget(i) == page)


class BFTabNavigation(QMainWindow):
    """Tab-based navigation interface."""

    def __init__(self):
        """Initialize tab-based navigation interface."""
        super().__init__()
        self.setWindowTitle("Navigation")
        self.setGeometry(100, 100, 800, 600)
        self.init_ui()

    def init_ui(self):
        """Initialize user interface components."""
        # Create tab widget with styles
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
                border-radius: 0;
                margin: 0;
            }
            QTabBar::tab:selected {
                background-color: #0288D1;
                border-bottom: 4px solid #0277BD;
            }
            QTabBar::tab:hover {
                background-color: #0277BD;
            }
            QTabBar::tab:!selected {
                margin-bottom: 2px;
            }
            QTabBar::tab:!selected:hover {
                background-color: #0277BD;
            }
            QTabBar {
                qproperty-drawBase: 0;
            }
            QTabBar::tab {
                width: 200px;
            }
        """)

        # Set as central widget
        self.setCentralWidget(self.tab_widget)

    def add_tab(self, tab_name, new_page):
        """
        Add tab to navigation interface.

        :param tab_name: Name to display on tab
        :type tab_name: str
        :param new_page: Page widget to display in tab
        :type new_page: QWidget
        """
        self.tab_widget.addTab(new_page, tab_name)


class BFLogWidget(QWidget):
    """Widget for displaying application logs and messages."""

    def __init__(self):
        """Initialize log display interface."""
        super().__init__()
        self.init_ui()

    def init_ui(self):
        """Initialize user interface components."""
        # Create main layout
        self.layout = QVBoxLayout()

        # Create log display label
        self.label = QLabel('Output Information')
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-weight: bold; font-size: 10pt;")

        # Create text display area
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

        # Create log management buttons
        button_layout = QHBoxLayout()
        self.save_button = QPushButton("Save Log")
        self.save_button.clicked.connect(self.save_log)
        self.clear_button = QPushButton("Clear Log")
        self.clear_button.clicked.connect(self.clear_log)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.clear_button)

        # Add components to layout
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.log_display)
        self.layout.addLayout(button_layout)

        # Set layout
        self.setLayout(self.layout)

    def add_log(self, event, is_error=False, is_success=False, is_start=False, is_end=False):
        """
        Add entry to log display.

        :param event: Log message text
        :type event: str
        :param is_error: Flag for error messages
        :type is_error: bool
        :param is_success: Flag for success messages
        :type is_success: bool
        :param is_start: Flag for process start messages
        :type is_start: bool
        :param is_end: Flag for process completion messages
        :type is_end: bool
        """
        # Format log entry
        current_time = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        if is_error:
            log_entry = f"<p style='margin-bottom: 15px; color: red;'><strong>{current_time}</strong><br><strong>[ERROR]</strong> {event}</p>"
        elif is_success:
            log_entry = f"<p style='margin-bottom: 15px; color: green;'><strong>{current_time}</strong><br><strong>[SUCCESS]</strong> {event}</p>"
        elif is_start:
            log_entry = f"<p style='margin-bottom: 15px; color: blue;'><strong>{current_time}</strong><br><strong>[START]</strong> {event}</p>"
        elif is_end:
            log_entry = f"<p style='margin-bottom: 15px; color: blue;'><strong>{current_time}</strong><br><strong>[FINISH]</strong> {event}</p>"
        else:
            log_entry = f"<p style='margin-bottom: 15px;'><strong>{current_time}</strong><br>{event}</p>"

        # Add to display
        self.log_display.append(log_entry)

    def save_log(self):
        """Save log content to file."""
        # Create log directory if needed
        log_folder = "log"
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        # Create timestamped filename
        current_time = QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss")
        log_filename = os.path.join(log_folder, f"log_{current_time}.txt")

        # Save log content
        with open(log_filename, "w", encoding='utf-8') as log_file:
            log_file.write(self.log_display.toPlainText())

    def clear_log(self):
        """Clear log display area."""
        self.log_display.clear()


class BFPanel(QGroupBox):
    """Styled container for grouping interface elements."""

    def __init__(self, title):
        """
        Initialize panel container.

        :param title: Panel title
        :type title: str
        """
        super().__init__(title)
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
            }
        """)
        self.vlayout = QVBoxLayout(self)


class BFGroupBox(QGroupBox):
    """Styled container for grouping interface elements."""

    def __init__(self, title):
        """
        Initialize group container.

        :param title: Container title
        :type title: str
        """
        super().__init__(title)
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
            }
        """)


class BFPushButton(QPushButton):
    """Styled push button with hover and press effects."""

    def __init__(self, text):
        """
        Initialize styled push button.

        :param text: Button text
        :type text: str
        """
        super().__init__(text)
        self.setStyleSheet(self.button_style())

    def button_style(self):
        """
        Define button styling.

        :return: CSS styling string
        :rtype: str
        """
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


class ProcessDialog(QDialog):
    """Dialog for displaying file processing status."""

    def __init__(self, import_file_list, parent=None):
        """
        Initialize file processing dialog.

        :param import_file_list: List of files to process
        :type import_file_list: list[str]
        :param parent: Parent widget
        :type parent: QWidget
        """
        super().__init__(parent)
        self.import_file_list = import_file_list
        self.init_ui()

    def init_ui(self):
        """Initialize user interface components."""
        self.setWindowTitle("Processing Files")
        self.resize(800, 300)

        # Create layout and file display tree
        layout = QVBoxLayout()
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["File", "Processed"])
        self.tree_widget.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        layout.addWidget(self.tree_widget)

        # Add file items to tree
        self.add_tree_items()
        self.setLayout(layout)

    def setTitle(self, text):
        """
        Set dialog title.

        :param text: Title text
        :type text: str
        """
        self.setWindowTitle(text)

    def add_tree_items(self):
        """Add files to display tree with initial status."""
        if not self.import_file_list:
            return

        # Handle single or multiple files
        if len(self.import_file_list) == 1:
            file_path = self.import_file_list[0]
            if not isinstance(file_path, str) and len(file_path) > 1:
                file_path = file_path[0]
            self.add_file_item(file_path)
        else:
            for file_path in self.import_file_list:
                if not isinstance(file_path, str) and len(file_path) > 1:
                    file_path = file_path[0]
                self.add_file_item(file_path)

        # Center-align tree items
        self.tree_widget.header().setDefaultAlignment(Qt.AlignCenter)

    def add_file_item(self, file_path):
        """
        Add file to display tree.

        :param file_path: Path of file to display
        :type file_path: str
        """
        item = QTreeWidgetItem([file_path, "No"])
        item.setTextAlignment(0, Qt.AlignCenter)
        item.setTextAlignment(1, Qt.AlignCenter)
        item.setForeground(1, QColor("red"))
        item.setFont(1, QFont("Arial", weight=QFont.Bold))
        self.tree_widget.addTopLevelItem(item)

    def update_status(self, index, status):
        """
        Update processing status for specific file.

        :param index: File index
        :type index: int
        :param status: New processing status
        :type status: str
        """
        item = self.tree_widget.topLevelItem(index)
        item.setText(1, status)
        if status == "No":
            item.setForeground(1, QColor("red"))
            item.setFont(1, QFont("Arial", weight=QFont.Bold))
        elif status == "Yes":
            item.setForeground(1, QColor("green"))
            item.setFont(1, QFont("Arial", weight=QFont.Bold))


class BFShowMessageEvent(QEvent):
    """Custom event for displaying application messages."""

    def __init__(self, message):
        """
        Initialize message event.

        :param message: Message to display
        :type message: str
        """
        super().__init__(QEvent.User)
        self.message = message


class BFFileListPanel(QWidget):
    """Interface for displaying and managing file lists."""

    def __init__(self):
        """Initialize file list panel."""
        super().__init__()
        self.file_list = []  # List of file paths
        self.file_status = {}  # Dictionary of file metadata
        self.show_files = {'processed': True, 'unprocessed': True}
        self.initUI()

    def initUI(self):
        """Initialize user interface components."""
        layout = QVBoxLayout()

        # Folder selection components
        folder_layout = QHBoxLayout()
        self.folder_button = BFPushButton("Select Folder")
        self.folder_button.clicked.connect(self.selectFolder)
        self.folder_lineedit = QLineEdit()
        self.folder_lineedit.setReadOnly(True)
        folder_layout.addWidget(self.folder_button)
        folder_layout.addWidget(self.folder_lineedit)
        layout.addLayout(folder_layout)

        # File display tree
        self.tree_widget = QTreeWidget()
        self.tree_widget.setColumnCount(4)
        self.tree_widget.setHeaderLabels(["File Name", "File Type", "Processed", "File Path"])
        self.tree_widget.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        for i in range(4):
            self.tree_widget.headerItem().setTextAlignment(i, Qt.AlignCenter)
        self.tree_widget.itemClicked.connect(self.printFilePath)
        layout.addWidget(self.tree_widget)

        self.setLayout(layout)

    def selectFolder(self):
        """Open folder selection dialog and display files."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.folder_lineedit.setText(folder)
            self.populateTree(folder)

    def populateTree(self, folder):
        """Populate file list from selected folder."""
        self.tree_widget.clear()
        self.file_list = []
        self.file_status = {}
        self.populateTreeRecursively(folder)
        self.showFiles()

    def populateTreeRecursively(self, folder):
        """Recursively process directory for files."""
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
                processed = "No"
                self.file_list.append(file_path)
                self.file_status[file_path] = {
                    "file_name": file_name,
                    "file_suffix": file_suffix,
                    "processed": processed
                }

    def showFiles(self):
        """Display files in tree widget based on visibility settings."""
        self.tree_widget.clear()
        for file_path in self.file_list:
            status = self.file_status[file_path]
            if (status["processed"] == "Yes" and self.show_files['processed']) or \
                    (status["processed"] == "No" and self.show_files['unprocessed']):
                item = QTreeWidgetItem([
                    status["file_name"],
                    status["file_suffix"],
                    status["processed"],
                    file_path
                ])
                for i in range(4):
                    item.setTextAlignment(i, Qt.AlignCenter)
                self.tree_widget.addTopLevelItem(item)

    def printFilePath(self, item, column):
        """
        Print file path when item is clicked.

        :param item: Clicked tree item
        :type item: QTreeWidgetItem
        :param column: Clicked column
        :type column: int
        """
        if column == 0:
            file_path = self.getFilePath(item)
            print("File Path:", file_path)

    def getFilePath(self, item):
        """
        Retrieve file path from tree item.

        :param item: Tree item
        :type item: QTreeWidgetItem
        :return: File path
        :rtype: str
        """
        return item.text(3)

    def get_files_by_extension(self, extensions):
        """
        Find files matching specified extensions.

        :param extensions: File extensions to match
        :type extensions: list[str]
        :return: Matching file paths
        :rtype: list[str]
        """
        return [file for file in self.file_list if file.split('.')[-1] in extensions]

    def updateShowFiles(self, processed=None, unprocessed=None):
        """
        Update visibility settings for files.

        :param processed: Show processed files
        :type processed: bool
        :param unprocessed: Show unprocessed files
        :type unprocessed: bool
        """
        if processed is not None:
            self.show_files['processed'] = processed
        if unprocessed is not None:
            self.show_files['unprocessed'] = unprocessed
        self.showFiles()

    def toggleProcessed(self, file_path):
        """
        Toggle processing status for file.

        :param file_path: Path of file to update
        :type file_path: str
        """
        if file_path in self.file_status:
            current_status = self.file_status[file_path]["processed"]
            new_status = "Yes" if current_status == "No" else "No"
            self.file_status[file_path]["processed"] = new_status
            self.showFiles()

    def showSpecifiedFiles(self, file_paths):
        """
        Display specified files only.

        :param file_paths: List of file paths to display
        :type file_paths: list[str]
        """
        self.tree_widget.clear()
        for file_path in file_paths:
            if file_path in self.file_status:
                status = self.file_status[file_path]
                item = QTreeWidgetItem([
                    status["file_name"],
                    status["file_suffix"],
                    status["processed"],
                    file_path
                ])
                for i in range(4):
                    item.setTextAlignment(i, Qt.AlignCenter)
                self.tree_widget.addTopLevelItem(item)


class BFParameterSwitchPanel(QWidget):
    """Panel for switching between different parameter sets."""

    def __init__(self):
        """Initialize parameter switching interface."""
        super().__init__()
        self.setWindowTitle("Parameter Switching App")
        self.setGeometry(100, 100, 800, 600)
        self.init_ui()

    def init_ui(self):
        """Initialize user interface components."""
        # Main layout
        main_layout = QVBoxLayout(self)

        # Parameter selection radio buttons
        self.radio_layout = QHBoxLayout()
        self.button_group = QButtonGroup()
        self.button_group.buttonClicked.connect(self.radio_button_clicked)

        # Parameter pages display
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

        # Add components to layout
        main_layout.addLayout(self.radio_layout)
        main_layout.addWidget(self.parameter_pages)
        main_layout.addStretch(1)

        # Storage for radio buttons and pages
        self.radio_dict = {}

    def add_radio_button_with_parameters(self, radio_text, parameter_widget, default=False):
        """
        Add radio button with associated parameter set.

        :param radio_text: Text for radio button
        :type radio_text: str
        :param parameter_widget: Widget to display when selected
        :type parameter_widget: QWidget
        :param default: Set as default selection
        :type default: bool
        """
        # Create radio button
        radio_button = QRadioButton(radio_text)
        self.radio_layout.addWidget(radio_button)

        # Create scrollable container for parameter widget
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(parameter_widget)

        # Add components to interface
        self.parameter_pages.addWidget(scroll_area)
        self.button_group.addButton(radio_button)
        self.radio_dict[radio_button] = scroll_area

        # Set default if specified
        if default:
            radio_button.setChecked(True)
            index = self.parameter_pages.indexOf(scroll_area)
            self.parameter_pages.setCurrentIndex(index)

    def radio_button_clicked(self, button):
        """
        Handle radio button selection.

        :param button: Selected radio button
        :type button: QRadioButton
        """
        index = self.parameter_pages.indexOf(self.radio_dict[button])
        self.parameter_pages.setCurrentIndex(index)


class BFSelectDialog(QDialog):
    """Dialog for selecting multiple items from a list."""

    def __init__(self, items, parent=None):
        """
        Initialize item selection dialog.

        :param items: Items to display for selection
        :type items: list[str]
        :param parent: Parent widget
        :type parent: QWidget
        """
        super(BFSelectDialog, self).__init__(parent)
        self.setWindowTitle("Select Items")
        self.init_ui(items)

    def init_ui(self, items):
        """Initialize user interface components."""
        self.layout = QVBoxLayout(self)

        # Create selectable item list
        self.listWidget = QListWidget(self)
        for item in items:
            listItem = QListWidgetItem(item)
            listItem.setFlags(listItem.flags() | Qt.ItemIsUserCheckable)
            listItem.setCheckState(Qt.Unchecked)
            self.listWidget.addItem(listItem)
        self.layout.addWidget(self.listWidget)

        # Add dialog buttons
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.layout.addWidget(self.buttonBox)

    def getSelectedItems(self):
        """
        Retrieve selected items.

        :return: List of selected item texts
        :rtype: list[str]
        """
        return [self.listWidget.item(index).text()
                for index in range(self.listWidget.count())
                if self.listWidget.item(index).checkState() == Qt.Checked]


class BFSelectWidget(QWidget):
    """Widget for selecting items from a list."""

    def __init__(self, text):
        """
        Initialize item selection widget.

        :param text: Button text
        :type text: str
        """
        super(BFSelectWidget, self).__init__()
        self.layout = QHBoxLayout(self)
        self.items = []
        self.init_ui(text)

    def init_ui(self, text):
        """Initialize user interface components."""
        # Create selection button
        self.button = BFPushButton(text)
        self.button.clicked.connect(self.openDialog)
        self.layout.addWidget(self.button)

        # Create display field
        self.lineEdit = QLineEdit(self)
        self.lineEdit.setReadOnly(True)
        self.layout.addWidget(self.lineEdit)

    def openDialog(self):
        """Open selection dialog and update display."""
        dialog = BFSelectDialog(self.items, self)
        if dialog.exec_() == QDialog.Accepted:
            selectedItems = dialog.getSelectedItems()
            self.lineEdit.setText(", ".join(selectedItems))

    def addItems(self, items):
        """
        Add items to selection list.

        :param items: Items to add to selection list
        :type items: list[str]
        """
        self.items = items

    def getSelectItem(self):
        """
        Retrieve selected items.

        :return: List of selected items
        :rtype: list[str]
        """
        return self.lineEdit.text().split(", ") if self.lineEdit.text() else []


class BFScrollArea(QScrollArea):
    """Scrollable container area for interface components."""

    def __init__(self):
        """Initialize scrollable container."""
        super(BFScrollArea, self).__init__()
        # Create container widget
        self.container_widget = QWidget()
        self.setWidget(self.container_widget)
        self.setWidgetResizable(True)
        self.setStyleSheet("QScrollArea { border: none; }")

    def set_layout(self, layout):
        """
        Set layout for container widget.

        :param layout: Layout to use for container
        :type layout: QLayout
        """
        self.container_widget.setLayout(layout)


# Application entry point
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication

    app = QApplication([])
    log_widget = BFFileListPanel()
    log_widget.show()
    sys.exit(app.exec_())