import threading

from PySide6.QtCore import *

class _WaiterQObject(QObject):
    finished = Signal()

    def __init__(self, delay):
        super().__init__()
        self.delay = delay
    
    def run(self):
        self.wait_event = threading.Event()
        self.wait_event.wait(timeout=self.delay)
        self.finished.emit()
    
    def stop(self):
        self.wait_event.set()

class Waiter:
    def __init__(self):
        self._waiter = None
        self._thread = None
        self._fun = None

    def wait_and_run(self, delay, fun):
        self._thread = QThread()
        self._waiter = _WaiterQObject(delay=delay)
        self._waiter.finished.connect(fun)
        self._waiter.moveToThread(self._thread)
        self._thread.started.connect(self._waiter.run)
        # self._waiter.finished.connect(self._thread.quit)
        # self._waiter.finished.connect(self._waiter.deleteLater)
        # self._thread.finished.connect(self._thread.deleteLater)
        self._fun = fun
        self._thread.start()

        # self._thread.finished.connect(fun)
    
    def quit_waiter(self):
        if self._thread is not None and self._waiter is not None:
            self._waiter.finished.disconnect(self._fun)
            self._waiter.stop()
            self._thread.quit()
            self._thread.wait()
            self._waiter.deleteLater()
            self._thread.deleteLater()

            self._fun = None
            self._waiter = None
            self._thread = None