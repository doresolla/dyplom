import re
import whisper
import operator
from yt_dlp import YoutubeDL
import subprocess

import math
import os


class AudioFile:
    """    """

    def __init__(self, s, duration, chapters):
        self.filename = s
        self.ext = self.filename[s.index('.'):]
        self.folder_name = self.filename[:s.index('.')] + '\\'
        self.abs_filename = f'C:\\Users\\dondu\\OneDrive\\Documents\\GitHub\\dyplom\\{self.filename}'
        self.duration = duration
        self.chapters = chapters
        self.create_folder()

    def create_folder(self):
        # создание папки для хранения исходного файла и его производных
        current_dir = os.path.dirname(os.path.realpath(__file__))
        dest_folder = os.path.join(current_dir, self.folder_name)
        if not os.path.isdir(dest_folder):
            os.mkdir(dest_folder)
            print(f"Создание директории для файла {self.filename}")
        elif not (os.path.isfile(dest_folder + self.filename)):
            # перемещение исходного файла и его аудио в созданную папку
            os.replace(self.abs_filename, dest_folder + self.filename)
            os.replace(current_dir+'\\tmp\\sub.srt.ru.vtt', dest_folder + '\\sub.srt.ru.vtt')
            self.abs_filename = dest_folder + self.filename
            print(f"Перемещение  файла {self.filename} в директорию {self.folder_name}")

    def convert_and_split(self, abs_filename):
        splits = []

        total_mins = math.ceil(self.duration / 60)
        counter = 0
        try:
            if (10 < total_mins):
                for i in range(0, total_mins, 5):
                    counter += 1
                    out = f'.\\{self.folder_name}\\{counter}_{self.filename}'
                    split_video(abs_filename, i, i + 5, out)
                #    convert_video_to_audio_ffmpeg(out[:out.rindex('.')], "wav")
                    splits.append(out)

            else:
                out = f'.\\{self.folder_name}\\{self.filename}'
                # convert_video_to_audio_ffmpeg(out[:out.rindex('.')], "wav")
                splits.append(out)
        except Exception as e:
            print("Ошибка при конвертировании и разделении")
            print(e)
        else:
            print("Конвертирование и разделение прошло успешно")
            return splits
    def recognizeSpeech(self, files, model):
        print(self.filename)
        start = self.duration //10
        current_dir = os.path.dirname(os.path.realpath(__file__))
        dest_folder = os.path.join(current_dir, self.folder_name)
        cut = dest_folder + '\\cut.wav'
        split_video(self.abs_filename,start, start + 30, cut )

        audio = whisper.load_audio(cut)
        audio = whisper.pad_or_trim(audio)
        #audio = path_to_audio
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)
        output = {k: d[k] for e, d in enumerate(sorted(
            probs, key=lambda d: next(d[k] for k in d), reverse=True)) for k in d if e < 3}

        print("Вероятности появления различных языков:", output )
        lang = max(output.items(), key=operator.itemgetter(1))[0]
        print ("Язык - ", lang)

        txt_file = self.filename[:self.filename.rindex('.')] + '.txt'
        for f in files:
            #текущий файл
            print('Start', f[f.rindex('\\') + 1 : ])
            result = model.transcribe(f, fp16=False, language=lang)

            # файл с расшифровкой речи
            # Если такого файла не существует
            if not (os.path.isfile(txt_file)):
                param = 'w'  # создать файл и записать в него
            else:
                param = 'a+'  # добавить данные в конец файла

            file = open(txt_file, param)
            file.write(result['text'] + '\n')
            file.close()
            print("Конец", f[f.rindex('\\') + 1:])


def download_audio(link, to_download):
    list_langs = []
    if link.__contains__('youtube'):
        with (YoutubeDL({
            'writeautomaticsub': True,
            'subtitlesformat': 'srt',
            'skip_download': True,
            'subtitleslangs':['ru'],
            'outtmpl': '/tmp/sub.srt'
        })) as ydl:
            if to_download:
                ydl.download(link)
            info_dict = ydl.extract_info(link, download=False)
            #разделение на главы (end_time, start_time, title)
            if ('chapters' in info_dict):
                chapters = info_dict['chapters']
            right_title = check_title(info_dict['title'])
        with (YoutubeDL({'extract_audio': True, 'format': 'bestaudio/best',
                         'postprocessors': [{
                             'key': 'FFmpegExtractAudio',
                             'preferredcodec': 'wav',
                             'preferredquality': '192',
                         }],
                         'outtmpl': right_title,
                         }) as YT):
            # информация о видео по ссылке
            if to_download:
                YT.download(link)
            info_dict = YT.extract_info(link, download=False)
        return right_title + '.wav', info_dict['duration'], chapters


def check_title(title):
    title = title.replace(' ', '_')
    title = re.sub(r'[^\w]', '_', title)
    title = re.sub(r'_{2,}', '_', title)
    title = re.sub(r'_$', '', title)

    return title


def convert_video_to_audio_ffmpeg(video_file, output_ext="mp3"):
    """Converts video to audio directly using `ffmpeg` command
    with the help of subprocess module"""
    filename, ext = os.path.splitext(video_file)

    subprocess.call(["ffmpeg", "-y", "-i", video_file, f"{filename}.{output_ext}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)


def split_video(file_path:str, start:int, end:int, output_path:str):
    ''' file_path  '''
    start_time = f'{start // 60}:{start % 60}:00'
    end_time = f'{end // 60}:{end % 60}:00'
    command = f"ffmpeg -i {file_path} -ss {start_time} -to {end_time} -c copy {output_path}"
    subprocess.call(command, shell=True)


def cut_video( filename, duration ):
    start_time = duration//10
    end_time = start_time + 30
    command = f"ffmpeg -i {file_path} -ss {start_time} -to {end_time} -c copy {output_path}"

    cmd =['ffmpeg','-i', ]

