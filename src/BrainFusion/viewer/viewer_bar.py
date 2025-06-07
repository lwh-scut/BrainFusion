import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


class MultiBarChartApp(QMainWindow):
    """
    Application for visualizing Excel data through multi-subplot bar charts.

    :param main_widget: Main container widget
    :type main_widget: QWidget
    :param canvas: Matplotlib figure canvas
    :type canvas: FigureCanvas
    """

    def __init__(self):
        """Initialize the main window components."""
        super().__init__()
        self.setWindowTitle("Excel Data Multi-Subplot Bar Chart Visualizer")
        self.setGeometry(100, 100, 1200, 800)

        # Main window layout components
        self.main_widget = QWidget()
        self.layout = QVBoxLayout(self.main_widget)

        # Excel file loader button
        self.button = QPushButton("Load Excel File")
        self.button.clicked.connect(self.load_excel_file)
        self.layout.addWidget(self.button)

        # Matplotlib visualization canvas
        self.canvas = FigureCanvas(plt.figure(figsize=(10, 8)))
        self.layout.addWidget(self.canvas)

        self.setCentralWidget(self.main_widget)

    def load_excel_file(self):
        """
        Open file dialog to select and load Excel data.

        :returns: None
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Excel File",
            "",
            "Excel Files (*.xlsx *.xls)"
        )
        if not file_path:
            return

        # Attempt to read Excel data
        try:
            data = pd.read_excel(file_path)
        except Exception as e:
            print(f"File loading failed: {e}")
            return

        # Validate data format
        if data.shape[1] < 2:
            print("Invalid data format - requires at least two columns")
            return

        # Generate bar charts
        self.plot_bar_charts(data)

    def plot_bar_charts(self, data):
        """
        Visualize Excel data through multi-subplot bar charts.

        :param data: Tabular data from Excel
        :type data: pandas.DataFrame
        """
        plt.clf()  # Clear existing figure

        # Calculate number of subplots (one per column pair)
        num_subplots = data.shape[1] // 2
        if num_subplots == 0:
            print("Insufficient columns - requires pairs of data columns")
            return

        # Create subplots
        fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 8))
        if num_subplots == 1:
            axes = [axes]  # Ensure axes is iterable for single subplot

        # Generate each subplot
        for i, ax in enumerate(axes):
            x = data.iloc[:, i * 2]  # First column in pair as x-values
            y = data.iloc[:, i * 2 + 1]  # Second column in pair as y-values

            # Create bar chart
            bars = ax.bar(x, y, width=0.6, color='skyblue', label=f'Chart {i + 1}')

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f'{height:.2f}',
                    ha='center',
                    va='bottom'
                )

            # Configure chart elements
            ax.legend()
            ax.set_title(f'Subplot {i + 1}')
            ax.set_xlabel("Category")
            ax.set_ylabel("Value")

        # Optimize layout spacing
        plt.tight_layout()

        # Refresh visualization
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MultiBarChartApp()
    window.show()
    sys.exit(app.exec_())