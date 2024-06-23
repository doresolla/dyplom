# -*- coding: utf-8
import os.path

import audio
from text import Text, read_subs
import whisper
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
        # если на компьютере нет такой директории link
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
                file = audio.AudioFile(filename, duration, chapters, False, self.link)
            except Exception as e:
                print(e)
                self.print_signal.result.emit("Ошибка: убедитесь, что файл доступен." + str(e))
        else:
            self.signals.result.emit("Статус: Скачивание видео")
            isThere = False
            filename, duration, chapters = audio.download_audio(self.link)
            print(chapters)
            self.signals.result.emit("Статус: Скачивание видео завершено")
            file = audio.AudioFile(filename, duration, chapters, isThere)

        if isinstance(file, audio.AudioFile):
            self.video.result.emit(file.filename)
            self.process_file(file)

    def process_file(self, file):
        start = file.duration // 10 // 60
        current_dir = path.dirname(path.realpath(__file__))
        dest_folder = path.join(current_dir, file.folder_name)
        cut = str(dest_folder + 'cut.wav')
        audio.split_video(file.folder_name + file.filename, start, start, cut)
        model = whisper.load_model("small", in_memory=True)
        cut = whisper.load_audio(cut)
        cut = whisper.pad_or_trim(cut)
        mel = whisper.log_mel_spectrogram(cut).to(model.device)
        _, probs = model.detect_language(mel)

        output = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        print("Вероятности появления различных языков:", output)
        lang = output[0]
        print("Язык - ", lang)
        # если есть неопределенность в языке
        if output[1][1] > 0.2 or lang[0] != 'ru':
            print(output[1])
            self.print_signal.result.emit("Поддержка других языков недоступна")
            return
        if not (path.isfile(file.folder_name+file.filename[:file.filename.rindex('.')] + '.txt')):
            try:
                # file.recognizeSpeech([file.folder_name + file.filename])
                self.signals.result.emit('Статус: идет распознавание речи, пожалуйста, подождите')
                file.recognizePywhisper_cpp()
            except Exception as e:
                print('Ошибка при распознавании голоса')
                print(e)
                self.print_signal.result.emit("Ошибка при распознавании голоса:" + str(e))
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
            # text_to_sum.text_rank()
        except Exception as e:
            self.print_signal.result.emit('Ошибка во время обработки текста ' + str(e))
            print(e)
        text_to_sum.add_data_export('dataset.csv')
        self.signals.result.emit("Статус: Работа завершена")

# def start_process(link):
#     # links = [  # 'https://vk.com/video-51126445_456243401',
#     #     'https://www.youtube.com/watch?v=k9wK2FThEsk&t',
#     #     'https://www.youtube.com/watch?v=I_ReFF3qiQ8',
#     #     # 'https://www.youtube.com/watch?v=jR6x5PmBL2I',
#     #     #  'https://www.youtube.com/watch?v=6dYPBA7-1Wg',
#     #     #   'https://www.youtube.com/watch?v=ML5tP8m6SHw',
#     #     #    'https://www.youtube.com/watch?v=k9 wK2FThEsk&t'
#     # ]
#     # for link in links:
