<<<<<<< HEAD
from main_window import Ui_MainWindow
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import QObject, pyqtSignal, QThreadPool
import sys, text, os

from mainAction import Main

class WorkerSignals(QObject):
    result = pyqtSignal(str)

=======
import sys


from PyQt6.QtCore import QObject, pyqtSignal, QThreadPool

from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
)

from main_window import Ui_MainWindow
from mainAction import Main


class WorkerSignals(QObject):
    result = pyqtSignal(str)


>>>>>>> feature/pipeline
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
<<<<<<< HEAD
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
=======

        self.status = WorkerSignals()
        self.status.result.connect(self.update_status_label)

        self.video_signal = WorkerSignals()
        self.video_signal.result.connect(self.update_videoname_label)

        self.print_signal = WorkerSignals()
        self.print_signal.result.connect(self.update_print_label)

        self._setup_model_selector()
        self.startButton.clicked.connect(self.start_thread)

    def _setup_model_selector(self):
        layout = QHBoxLayout()

        self.label_model = QLabel("Модель конспекта", parent=self.centralwidget)
        self.comboModel = QComboBox(parent=self.centralwidget)

        self.comboModel.addItem("Qwen", "qwen")
        self.comboModel.addItem("Mistral", "mistral")

        layout.addWidget(self.label_model)
        layout.addWidget(self.comboModel)

        # Вставляем строку выбора модели после строки ввода ссылки/пути
        self.verticalLayout.insertLayout(1, layout)

    def start_thread(self):
        try:
            input_value = self.lineEdit.text().strip().strip('"')
            if input_value:
                selected_model = self.comboModel.currentData()
                main_process = Main(
                    signal=self.status,
                    link=input_value,
                    set_videoname_signal=self.video_signal,
                    print_signal=self.print_signal,
                    llm_model=selected_model,
                )
                self.label_status.setText("Статус: Начало работы")
                self.textPrint.setText("")
                QThreadPool.globalInstance().start(main_process)
            else:
                self.textPrint.setText("Введите корректную ссылку на видео или путь к файлу")
        except Exception as e:
            current_text = self.textPrint.toPlainText()
            if current_text:
                self.textPrint.setText(current_text + "\n" + str(e))
            else:
                self.textPrint.setText(str(e))

    def update_status_label(self, text):
        self.label_status.setText(text)

    def update_videoname_label(self, text):
        self.label_videoname.setText(f'Название видео: "{text}"')

    def update_print_label(self, text):
        current_text = self.textPrint.toPlainText()
        if current_text:
            self.textPrint.setText(current_text + "\n" + text)
        else:
            self.textPrint.setText(text)


>>>>>>> feature/pipeline
def show_exception_and_exit(exc_type, exc_value, tb):
    import traceback
    traceback.print_exception(exc_type, exc_value, tb)
    input("Press key to exit.")
    sys.exit(-1)

<<<<<<< HEAD
if __name__ == '__main__':

=======

if __name__ == "__main__":
>>>>>>> feature/pipeline
    sys.excepthook = show_exception_and_exit
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
<<<<<<< HEAD
    sys.exit(app.exec())

=======
    sys.exit(app.exec())
>>>>>>> feature/pipeline
