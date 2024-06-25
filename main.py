from main_window import Ui_MainWindow
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import QObject, pyqtSignal, QThreadPool
import sys, text, os

from mainAction import Main

class WorkerSignals(QObject):
    result = pyqtSignal(str)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.status = WorkerSignals()
        self.status.result.connect(self.update_status_label)
        self.video_signal = WorkerSignals()
        self.video_signal.result.connect(self.update_videoname_label)
        self.print_signal = WorkerSignals()
        self.print_signal.result.connect(self.update_print_label)

        self.startButton.clicked.connect(self.start_thread)


    def start_thread(self):
        try:
            text = self.lineEdit.text().strip('"')
            if text != '':
                main_process = Main(self.status, text, self.video_signal, self.print_signal)
                self.label_status.setText("Начало работы")
                QThreadPool.globalInstance().start(main_process)
                self.textPrint.setText("")
            else:
                self.textPrint.setText('Введите корректное значение ссылки на видео')
        except Exception as e:
            text = self.textPrint.toPlainText()
            self.textPrint.setText(text + '\n' + e)

    def update_status_label(self, text):
        self.label_status.setText(text)
        # if text == 'Работа завершена':
        #     self.textPrint.setText()
    def update_videoname_label(self, text):
        self.label_videoname.setText(f'Название видео: \"{text}')
        # if text == 'Работа завершена':
        #     self.textPrint.setText()

    def update_print_label(self, text):
        textPrint = self.textPrint.toPlainText()
        if textPrint != '':
            self.textPrint.setText(textPrint + '\n' + text)
        else:
            self.textPrint.setText(text)
def show_exception_and_exit(exc_type, exc_value, tb):
    import traceback
    traceback.print_exception(exc_type, exc_value, tb)
    input("Press key to exit.")
    sys.exit(-1)

if __name__ == '__main__':

    sys.excepthook = show_exception_and_exit
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

