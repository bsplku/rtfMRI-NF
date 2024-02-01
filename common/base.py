import textwrap
from typing import Dict, List

import matplotlib.pyplot as plt

from PySide6.QtCore import *
from PySide6.QtWidgets import *
from PySide6.QtMultimedia import (QAudioOutput, QMediaPlayer)
from PySide6.QtMultimediaWidgets import QVideoWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class TaskMainWindow(QWidget):
    def __init__(self, parent: QMainWindow, task_params: Dict):
        super().__init__()
        self.experiment_main_window = parent
        self.params = task_params

        self._layout = QVBoxLayout(self)
        
        self.setStyleSheet('background-color: black;')
        self.setGeometry(5000, 2000, 800, 600)

        self.fig = plt.figure()
        self.fig.tight_layout(pad=0)
        self.fig.patch.set_facecolor('black') # set figure (not subplot) background as black
        self.canvas = FigureCanvas(self.fig)
        self._layout.addWidget(self.canvas, stretch=1)
        self.setLayout(self._layout)

        self.pages: List[TaskPage] = None
        self.current_page: TaskPage = None

        self.set_task_parameters()
        self.init_ui()
        self.load_resource()
        self.load_page()
    
    def set_task_parameters(self):
        self.current_mri_volume = 0
    
    def init_ui(self):
        '''
        method to load task pages
        '''
        raise NotImplementedError('Need to implement init_ui().')

    def load_resource(self):
        pass

    def unload_resource(self):
        pass

    def _get_page(self):
        self.current_page = self.pages.pop(0)

    def _setup_page(self):
        self.current_page.init_ui()
        self.current_page.load_resource()
        self.current_page.load_page()

    def _cleanup_page(self):
        self.current_page.unload_resource()
        self.current_page.unload_page()
        self.current_page.deleteLater()
        self.current_page = None
    
    def load_page(self):
        if self.current_page is None:
            self._get_page()
            # set the first page.
            self._setup_page()
            self.fig.tight_layout(pad=0)
    
    def next_page(self):
        if len(self.pages) > 0 and self.current_page is not None:
            # clean-up the previous page widget.
            self._cleanup_page()

            self._get_page()
            # set next page.
            self._setup_page()
    
    def closeEvent(self, e):
        self._cleanup_page()
        self.unload_resource()
        self.hide()
        super().closeEvent(e)
    
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.unload_resource()
            self.close()
    
        elif e.key() == Qt.Key_F:
            self.fig.tight_layout(pad=0)
            self.showFullScreen()

        elif e.key() == Qt.Key_N:
            self.showNormal()
        
        elif e.key() == Qt.Key_S:
            self.handle_sync()
        
        self.current_page.keyPressEvent(e)
    
    def handle_sync(self):
        self.current_mri_volume += 1

    def clear_figure(self):
        self.fig.clear()
    
    def set_axis(self, spines=False, ticks=False):
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        # remove spines
        if not spines:
            ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)

        # remove ticks
        if not ticks:
            ax.set_xticks([])
            ax.set_yticks([])

        return ax
    
    def show_phrase(self, content, bg_color='black', font_color='white', size=32):
        font_option = {'color': font_color, 'size': size}

        ax = self.set_axis()

        # set background color
        ax.set_facecolor(bg_color)

        # place text
        txt = ax.text(0.5, 0.5, '\n'.join(textwrap.wrap(content, 30, break_long_words=False, replace_whitespace=False)), fontdict=font_option,\
                      horizontalalignment='center', verticalalignment='center')
        txt.set_linespacing(2)

        self.fig.tight_layout(pad=0)
        self.canvas.draw()

class TaskPage(QWidget):
    def __init__(self, parent: TaskMainWindow):
        super().__init__()
        self.task_main_window = parent
    
    def init_ui(self):
        raise NotImplementedError('Need to implement init_ui().')

    def load_resource(self):
        pass

    def unload_resource(self):
        pass

    def load_page(self):
        self.setFocus()

    def unload_page(self):
        pass
    
    def next_page(self):
        self.task_main_window.next_page()
    
    def keyPressEvent(self, e):
        pass
