# -*- coding: utf-8
import os.path

from . import audio
from .text import Text
# import whisper
from os import path
from PyQt6.QtCore import QRunnable


class Main(QRunnable):
    def __init__(self, signal, link, set_videoname_signal, print_signal):
        super().__init__()
        self.signals = signal
        self.video = set_videoname_signal
        self.print_signal = print_signal
        self.link = link.strip('"')

    def run(self):
        file = ''
        # если на компьютере такая директория link
        if path.exists(path.dirname(self.link)):
            try:
                filename = self.link[self.link.rindex('\\') + 1:]
                print(filename)
                duration = audio.get_length(self.link)
                chapters = []
                dict = {}
                if (os.path.isfile(filename[:filename.rindex('.')] + '\\chapters.txt')):
                    with open(filename[:filename.rindex('.')] + '\\chapters.txt', 'r', encoding='utf-8') as f:
                        s = f.readline()
                        dict['start_time'], dict['title'], dict['end_time'] = s.split('\t')
                        chapters.append(dict)
                audio.convert_video_to_audio_ffmpeg(self.link)
                file = audio.AudioFile(filename, duration, chapters, False, self.link)
            except Exception as e:
                print(e)
                self.signals.result.emit("Ошибка: убедитесь, что файл доступен")
                self.print_signal.result.emit(str(e))
        else:
            self.signals.result.emit("Статус: Скачивание видео")
            isThere = False
            try:
                filename, duration, chapters = audio.download(self.link)
                print(chapters)
                self.signals.result.emit("Статус: Скачивание видео завершено")
                file = audio.AudioFile(filename, duration, chapters, isThere)
            except Exception as e:
                print(e)

        if isinstance(file, audio.AudioFile):
            self.video.result.emit(file.filename[:file.filename.rindex('.')])
            self.process_file(file)

    def process_file(self, file):
        start = file.duration // 10 // 60
        current_dir = path.dirname(path.realpath(__file__))
        dest_folder = path.join(current_dir, file.folder_name)
        cut = str(dest_folder + 'cut.wav')
        audio.split_video(file.folder_name + file.filename, start, start, cut)
        # model = whisper.load_model("small", in_memory=True)
        # cut = whisper.load_audio(cut)
        # cut = whisper.pad_or_trim(cut)
        # mel = whisper.log_mel_spectrogram(cut).to(model.device)
        # _, probs = model.detect_language(mel)
        #
        # output = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        # print("Вероятности появления различных языков:", output)
        # lang = output[0]
        # print("Язык - ", lang)
        # # если есть неопределенность в языке
        # if output[1][1] > 0.2 or lang[0] != 'ru':
        #     print(output[1])
        #     self.print_signal.result.emit("Поддержка других языков недоступна")
        #     return
        if not (path.isfile(file.folder_name+file.filename[:file.filename.rindex('.')] + '.txt')):
            try:
                # file.recognizeSpeech([file.folder_name + file.filename])
                self.signals.result.emit('Статус: идет распознавание речи, пожалуйста, подождите')
                file.recognizePywhisper_cpp()
            except Exception as e:
                print('Ошибка при распознавании голоса')
                print(e)
                self.signals.result.emit("Статус: Ошибка при распознавании голоса")
                self.print_signal.result.emit(str(e))
            else:
                print("Распознавание прошло успешно")
                self.signals.result.emit("Статус: Распознавание прошло успешно")
        # file.extract_image()
        name = file.filename[:file.filename.index('.')]
        self.signals.result.emit("Статус: Выполняется обработка текста")
        text_to_sum = Text(name, file.chapters)
        try:
            text_to_sum.sent_summary()
            text_to_sum.sumy_sum()
            # # text_to_sum.text_rank()
            # text_to_sum.compare_sum()
        except Exception as e:
            self.signals.result.emit('Статус: Ошибка во время обработки текста ')
            self.print_signal.result.emit(str(e))
            print(e)
        # text_to_sum.add_data_export('dataset.csv')
        else:
            self.signals.result.emit("Статус: Работа завершена")
            self.print_signal.result.emit(f"Конспекты для видео \"{text_to_sum.name}\" сохранены. Пожалуйста, проверьте директорию файла")

def generate_summary(url, video_id, format='docx', ratio=0.5):
    summary_path = ''
    file = ''
    error_message = None
    # если на компьютере такая директория link
    if path.exists(path.dirname(url)):
        try:
            filename = os.path.basename(url)
            print(f'ID-{video_id}: Название видео:{filename}, {url}')
            chapters = []
            dict = {}
            chapters_path = os.path.join(os.path.dirname(url), filename[:filename.rindex('.')] + '_chapters.txt')
            if (os.path.isfile(chapters_path)):
                with open(chapters_path, 'r', encoding='utf-8') as f:
                    s = f.readline()
                    dict['start_time'], dict['title'], dict['end_time'] = s.split('\t')
                    chapters.append(dict)
                print(f'ID-{video_id}: Скачаны chapters')
            audio.convert_video_to_audio_ffmpeg(url)
            print(f'ID-{video_id}: Видео {filename} конвертировано в аудио')
            file = audio.AudioFile(url, chapters, False, url)
        except Exception as e:
            error_message = f"ID-{video_id}:Ошибка: убедитесь, что файл доступен {e}"
            print(error_message)
            return None, error_message
    # else:
    #     print("Статус: Скачивание видео")
    #     isThere = False
    #     try:
    #         filename, duration, chapters = audio.download(url)
    #         print(chapters)
    #         print("Статус: Скачивание видео завершено")
    #         file = audio.AudioFile(filename, duration, chapters, isThere, video_id=video_id)
    #     except Exception as e:
    #         print(e)
    #         audio.report_error(video_id, e)

    if isinstance(file, audio.AudioFile):
        print(f'ID-{video_id}:{os.path.splitext(file.filename)[0]}')
        summary_path, error_message = process_file(file, format, video_id, ratio)

        print(f'ID-{video_id}: filename={file.filename}')
        os.rename(summary_path, file.filename)

    return summary_path, error_message
def process_file(file, format, video_id, ratio=0.5):
    # start = file.duration // 10 // 60
    # current_dir = path.dirname(path.realpath(__file__))
    # dest_folder = path.join(current_dir, file.folder_name)
    # cut = str(dest_folder + 'cut.wav')
    # audio.split_video(file.folder_name + file.filename, start, start, cut)
    # model = whisper.load_model("small", in_memory=True)
    # cut = whisper.load_audio(cut)
    # cut = whisper.pad_or_trim(cut)
    # mel = whisper.log_mel_spectrogram(cut).to(model.device)
    # _, probs = model.detect_language(mel)
    #
    # output = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    # print("Вероятности появления различных языков:", output)
    # lang = output[0]
    # print("Язык - ", lang)
    # если есть неопределенность в языке
    # if output[1][1] > 0.2 or lang[0] != 'ru':
    #     print(output[1])
    #     print("Поддержка других языков недоступна")
    #     return
    name = os.path.splitext(file.filename)[0]
    txt_filename = name + '.txt'
    txt_path = os.path.join(file.folder_name, txt_filename)
    if not (path.isfile(txt_path)):
        try:
            print(f'ID-{video_id}: Идет распознавание речи, пожалуйста, подождите')
            file.recognizePywhisper_cpp()
        except Exception as e:
            error_message = f'ID-{video_id}: Ошибка при распознавании голоса {e}'
            print(error_message)
            return None, error_message
        else:
            print(f"ID-{video_id}:Распознавание прошло успешно")
    print(f"ID-{video_id}: Выполняется обработка текста")
    text_to_sum = Text(file.abs_filename, file.chapters,SEN_PERCENT=ratio)
    try:
        text_to_sum.sent_summary()
        filename_pathes = text_to_sum.sumy_sum(format)
        text_to_sum.compare_sum()
        print(f"ID-{video_id}: Работа завершена")
        print(f"ID-{video_id}: Конспекты для видео \"{text_to_sum.name}\" сохранены")
        ### TODO : выбор алгоритма суммаризации
        return filename_pathes[0], None
    except Exception as e:
        error_message = f'ID-{video_id}: Ошибка во время обработки текста {e}'
        print(error_message)
        return None, error_message
        # audio.report_error(video_id, e)
    # text_to_sum.add_data_export('dataset.csv')