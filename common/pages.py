from PySide6.QtGui import *
from PySide6.QtCore import *
from PySide6.QtWidgets import *

from .base import TaskMainWindow, TaskPage
from .resources import Waiter

class SyncWaitPage(TaskPage):
    def init_ui(self):
        self.task_main_window.show_phrase('Waiting for Sync')
    
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_S:
            self.task_main_window.next_page()

class ReadyPage(TaskPage):
    def __init__(self, parent: TaskMainWindow):
        super().__init__(parent)
        self.delay = None
        self.init_delay()
        if self.delay is None:
            raise ValueError('self.delay was not set.')
    
    def init_delay(self):
        raise NotImplementedError('Need to implement init_delay().')

    def init_ui(self):
        self.task_main_window.show_phrase('Ready')
    
    def load_resource(self):
        self.waiter = Waiter()
    
    def unload_resource(self):
        self.waiter.quit_waiter()
        self.waiter = None
    
    def load_page(self):
        super().load_page()
        self.waiter.wait_and_run(self.delay, self.next_page)

class FixationPage(TaskPage):
    def __init__(self, parent: TaskMainWindow):
        super().__init__(parent)
        self.delay = None
        self.init_delay()
        if self.delay is None:
            raise ValueError('self.delay was not set.')
    
    def init_delay(self):
        raise NotImplementedError('Need to implement init_delay().')

    def init_ui(self):
        self.task_main_window.show_phrase('+')
    
    def load_resource(self):
        self.waiter = Waiter()
    
    def unload_resource(self):
        self.waiter.quit_waiter()
        self.waiter = None
    
    def load_page(self):
        super().load_page()
        self.waiter.wait_and_run(self.delay, self.next_page)

class EndPage(TaskPage):
    def init_ui(self):
        self.task_main_window.show_phrase('End')