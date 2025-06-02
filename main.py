import urllib.parse

from main_window import Ui_MainWindow
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import QObject, pyqtSignal, QThreadPool
import sys, requests, time, os
from mainAction import Main, generate_summary

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

def download_file(url, dest_path):
    print(f"Скачиваю {url}")
    r = requests.get(url, stream=True)
    r.raise_for_status()

    with open(dest_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
def report_error(video_id, message):
    res = requests.post(
        "https://web-production-fdfb.up.railway.app/api/report_error/",
        data={"id": video_id, "message": message}
    )
    print(f"Ошибка сохранена: {res.status_code}")

if __name__ == '__main__':
    SERVER_URL = "https://web-production-fdfb.up.railway.app"
    CHECK_INTERVAL = 30  # секунд

    while True:
        try:
            print("Проверяю наличие новых заданий...")
            response = requests.get(f"{SERVER_URL}/api/pending/")
            response.raise_for_status()
            tasks = response.json()
            for task in tasks:
                print("Получено видео")
                video_url = SERVER_URL + task["video_url"]
                parsed_path = urllib.parse.urlparse(video_url).path  # → "/media/videos/15%D1%81%D0%BA.mp4"
                # 2. Декодируем URL-часть
                decoded_path = urllib.parse.unquote(parsed_path)  # → "/media/videos/15сек.mp4"

                # 3. Извлекаем только имя файла
                filename = os.path.basename(decoded_path)  # → "15сек.mp4"
                dirname = filename[:filename.rindex('.')]

                os.makedirs(dirname, exist_ok=True)  # создаёт папку, если её нет
                id = task["id"]
                format = task["format"]
                ratio = task["ratio"]
                print(video_url)
                if os.path.exists(dirname+'\\'+filename):
                    results = [f'{dirname}\\{x}.{format}' for x in ['lsa','lex_rank','text_rank_sumy','luhn'] ]

                    for summary_path in results:
                        upload = requests.post("https://web-production-fdfb.up.railway.app/api/upload_summary/",
                                               data={"id": id},
                                               files={"summary_file": open(summary_path, "rb")})
                        print("Ответ сервера:", upload.status_code, upload.text)
                else:
                    try:
                        download_file(video_url, dirname+'\\'+filename)
                        if os.path.exists(dirname+'\\'+filename):
                            print("Начало работы")
                            results = generate_summary(dirname+'\\'+filename, video_id=id, format=format, ratio=ratio)

                            for summary_path in results:
                                upload = requests.post("https://web-production-fdfb.up.railway.app/api/upload_summary/",
                                                       data={"id": id},
                                                       files={"summary_file": open(summary_path, "rb")})
                                print("Ответ сервера:", upload.status_code, upload.text)
                    except Exception as e:
                        print("Ошибка:", e)
                        report_error(id, e)
        except Exception as e:
            print("Ошибка:", e)
        print(f"Жду {CHECK_INTERVAL} секунд...\n")
        time.sleep(CHECK_INTERVAL)
    # sys.excepthook = show_exception_and_exit
    # app = QApplication(sys.argv)
    # window = MainWindow()

    # sys.exit(app.exec())

# python manage.py makemigrations  &&
# python manage.py showmigrations &&
# python manage.py migrate website &&
# python manage.py collectstatic --noinput &&
# python manage.py migrate sessions &&
# gunicorn trpo.wsgi:application --timeout 90 --bind 0.0.0.0:$PORT