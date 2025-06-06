import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ScatterPlotWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scatter Plot in PyQt5")
        self.setGeometry(100, 100, 800, 600)

        # 创建中央小部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Matplotlib 图表嵌入
        self.canvas = FigureCanvas(Figure(figsize=(8, 6)))
        layout.addWidget(self.canvas)

    def plot_scatter(self, data_dict):
        ax = self.canvas.figure.add_subplot(111)
        ax.clear()

        # 绘制散点图
        for group, values in data_dict.items():
            ax.scatter([group] * len(values), values, label=group)

        # 图形设置
        ax.set_title("Scatter Plot by Group")
        ax.set_xlabel("Groups")
        ax.set_ylabel("Values")
        ax.legend()
        self.canvas.draw()


import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class DensityHistogramWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Density Histogram in PyQt5")
        self.setGeometry(100, 100, 800, 600)

        # 创建中央小部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Matplotlib 图表嵌入
        self.canvas = FigureCanvas(Figure(figsize=(8, 6)))
        layout.addWidget(self.canvas)

    def plot_density_histogram(self, data_dict):
        ax = self.canvas.figure.add_subplot(111)
        ax.clear()

        # 绘制密度直方图
        for group, values in data_dict.items():
            values = np.array(values)  # 确保数据是 NumPy 数组
            ax.hist(
                values, bins=20, alpha=0.5, density=True, label=f"{group} (Density)"
            )

        # 图形设置
        ax.set_title("Density Histogram by Group")
        ax.set_xlabel("Values")
        ax.set_ylabel("Density")
        ax.legend()
        self.canvas.draw()


import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class SignificanceBoxPlotWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Box Plot with Significance")
        self.setGeometry(100, 100, 800, 600)

        # 创建中央小部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Matplotlib 图表嵌入
        self.canvas = FigureCanvas(plt.figure(figsize=(10, 6)))
        layout.addWidget(self.canvas)

    def plot_boxplot_with_significance(self, data_dict, significance_dict):
        ax = self.canvas.figure.add_subplot(111)
        ax.clear()

        # 准备数据
        groups = list(data_dict.keys())
        data = [data_dict[group] for group in groups]
        positions = np.arange(len(groups))  # 每组的位置

        # 绘制箱线图
        boxplot = ax.boxplot(
            data,
            positions=positions,
            patch_artist=True,
            notch=True
        )

        # 设置每组的颜色
        colors = plt.cm.tab10.colors  # 默认颜色
        for patch, color in zip(boxplot['boxes'], colors[:len(groups)]):
            patch.set_facecolor(color)

        # 添加显著性标记
        significance_level = {0.0001: '***', 0.001: '**', 0.05: '*'}
        bracket_height = 0.2  # 括号的高度
        asterisk_offset = 0.01  # 星号相对于括号的偏移
        line_width = 1.5  # 横线宽度

        for (group1, group2), p_value in significance_dict.items():
            if p_value < max(significance_level.keys()):
                # 确定显著性星号
                significance_marker = next(v for k, v in significance_level.items() if p_value < k)

                # 获取组的位置
                index1, index2 = groups.index(group1), groups.index(group2)
                x1, x2 = positions[index1], positions[index2]
                max_value = max(max(data_dict[group1]), max(data_dict[group2]))
                y_bracket = max_value + bracket_height
                y_asterisk = y_bracket + asterisk_offset

                # 在两个箱线图之间绘制星号
                ax.text((x1 + x2) / 2, y_asterisk, significance_marker, ha='center', va='bottom', fontsize=14,
                        color='red')

                # 在星号下方绘制横线括号
                ax.plot([x1, x2], [y_bracket, y_bracket], color='gray', lw=line_width)
                ax.plot([x1, x1], [y_bracket, y_bracket - 0.06], color='gray', lw=line_width)
                ax.plot([x2, x2], [y_bracket, y_bracket - 0.06], color='gray', lw=line_width)

        # 或者你可以在图形下方添加额外的说明文字（例如，如何读取显著性标记等）
        ax.text(0.5, -0.2, 'Significance markers: * p < 0.05, ** p < 0.001, *** p < 0.0001',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)

        # 图形设置
        ax.set_title("Box Plot with Significance")
        ax.set_xlabel("Groups")
        ax.set_ylabel("Values")
        ax.set_xticks(positions)
        ax.set_xticklabels(groups)

        self.canvas.draw()


from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class SignificanceViolinPlotWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Violin Plot with Significance")
        self.setGeometry(100, 100, 800, 600)

        # 创建中央小部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Matplotlib 图表嵌入
        self.canvas = FigureCanvas(plt.figure(figsize=(10, 6)))
        layout.addWidget(self.canvas)

    def plot_violin_with_significance(self, data_dict, significance_dict):
        ax = self.canvas.figure.add_subplot(111)
        ax.clear()

        # 准备数据
        groups = list(data_dict.keys())
        data = [data_dict[group] for group in groups]
        positions = np.arange(len(groups))  # 每组的位置

        # 绘制小提琴图
        violin_parts = ax.violinplot(data, positions=positions, showmedians=True, showmeans=False, showextrema=False)

        # 设置每组的颜色
        colors = plt.cm.tab10.colors  # 默认颜色
        for pc, color in zip(violin_parts['bodies'], colors[:len(groups)]):
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)  # 调整透明度

        # 添加显著性标记
        significance_level = {0.0001: '***', 0.001: '**', 0.05: '*'}
        bracket_height = 0.2  # 括号的高度
        asterisk_offset = 0.01  # 星号相对于括号的偏移
        line_width = 1.5  # 横线宽度

        for (group1, group2), p_value in significance_dict.items():
            if p_value < max(significance_level.keys()):
                # 确定显著性星号
                significance_marker = next(v for k, v in significance_level.items() if p_value < k)

                # 获取组的位置
                index1, index2 = groups.index(group1), groups.index(group2)
                x1, x2 = positions[index1], positions[index2]
                max_value = max(max(data_dict[group1]), max(data_dict[group2]))
                y_bracket = max_value + bracket_height
                y_asterisk = y_bracket + asterisk_offset

                # 在两个小提琴图之间绘制星号
                ax.text((x1 + x2) / 2, y_asterisk, significance_marker, ha='center', va='bottom', fontsize=14,
                        color='red')

                # 在星号下方绘制横线括号
                ax.plot([x1, x2], [y_bracket, y_bracket], color='gray', lw=line_width)
                ax.plot([x1, x1], [y_bracket, y_bracket - 0.06], color='gray', lw=line_width)
                ax.plot([x2, x2], [y_bracket, y_bracket - 0.06], color='gray', lw=line_width)

        # 或者你可以在图形下方添加额外的说明文字（例如，如何读取显著性标记等）
        ax.text(0.5, -0.2, 'Significance markers: * p < 0.05, ** p < 0.001, *** p < 0.0001',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)

        # 图形设置
        ax.set_title("Violin Plot with Significance")
        ax.set_xlabel("Groups")
        ax.set_ylabel("Values")
        ax.set_xticks(positions)
        ax.set_xticklabels(groups)

        self.canvas.draw()


if __name__ == "__main__":
    # 示例数据
    np.random.seed(42)
    data_dict = {
        "Group 1": np.random.normal(5, 1, 30),
        "Group 2": np.random.normal(6, 1, 30),
        "Group 3": np.random.normal(7, 1, 30),
        "Group 4": np.random.normal(7, 1, 30),
        "Group 5": np.random.normal(7, 1, 30),
    }

    significance_dict = {
        ("Group 1", "Group 2"): 0.03,
        ("Group 2", "Group 3"): 0.0005,
        ("Group 1", "Group 3"): 0.2,
        ("Group 1", "Group 4"): 0.00002,
        ("Group 1", "Group 5"): 0.00002,
    }

    # 启动 PyQt5 应用程序
    app = QApplication(sys.argv)
    main_window = SignificanceViolinPlotWindow()
    main_window.plot_violin_with_significance(data_dict, significance_dict)
    main_window.show()
    sys.exit(app.exec_())
