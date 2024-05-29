import re
import time
from yt_dlp import YoutubeDL
import subprocess
from math import ceil
import os


class AudioFile:
    """    """

    def __init__(self, s, duration, chapters):
        self.filename = s
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
        if not (os.path.isfile(dest_folder + self.filename)):
            # перемещение исходного файла и его аудио в созданную папку
            os.replace(self.abs_filename, dest_folder + self.filename)
            os.replace(self.abs_filename[:self.abs_filename.index('.')] + '.mp4', dest_folder + self.filename[:self.filename.index('.')] + '.mp4')
            os.replace(current_dir + '\\tmp\\sub.srt.ru.vtt', dest_folder + '\\sub.srt.ru.vtt')
            self.abs_filename = dest_folder + self.filename
            print(f"Перемещение  файла {self.filename} в директорию {self.folder_name}")

    def split(self, abs_filename):
        splits = []
        total_mins = ceil(self.duration / 60)
        counter = 0
        try:
            if (10 < total_mins):
                for i in range(0, total_mins, 5):
                    counter += 1
                    out = f'.\\{self.folder_name}\\{counter}_{self.filename}'
                    split_video(abs_filename, i, i + 5, out)
                    splits.append(out)

            else:
                out = f'.\\{self.folder_name}\\{self.filename}'
                splits.append(out)
        except Exception as e:
            print("Ошибка при разделении")
            print(e)
        else:
            print("Разделение прошло успешно")
            return splits

    def recognizeSpeech(self, files, model):
        # файл с расшифровкой речи
        txt_file = self.filename[:self.filename.rindex('.')] + '.txt'
        # Если такого файла не существует
        start_time = time.time()
        for f in files:
            # текущий файл
            print('Start', f[f.rindex('\\') + 1:])
            result = model.transcribe(f, fp16=False, language='ru')
            if not (os.path.isfile(self.folder_name + txt_file)):
                param = 'w'  # создать файл и записать в него
            else:
                param = 'a+'  # добавить данные в конец файла
            with open(self.folder_name+ txt_file, param,  encoding="utf-8") as file:
                file.write(result['text'] + '\n')
            print("Конец", f[f.rindex('\\') + 1:])
        print("--- %s seconds ---" % (time.time() - start_time))

def download_audio(link: str, to_download=True):
    try:
        with (YoutubeDL({
            'writeautomaticsub': True,
            'subtitlesformat': 'srt',
            'skip_download': True,
            'subtitleslangs': ['ru'],
            'outtmpl': '/tmp/sub.srt'
        })) as ydl:
            if to_download:
                ydl.download(link)
            info_dict = ydl.extract_info(link, download=False)
            # разделение на главы (end_time, start_time, title)
            if ('chapters' in info_dict):
                chapters = info_dict['chapters']
            right_title = check_title(info_dict['title'])
        with (YoutubeDL({
            # 'format': 'bestvideo[ext=mp4][height<=720]+bestaudio[ext=wav]/mp4',
            'outtmpl': '{}.%(ext)s'.format(right_title),  # Имя файла будет основано на названии видео
            'merge_output_format': 'mp4',  # Формат выходного файла
        }) as YT):

            # информация о видео по ссылке
            if to_download:
                YT.download(link)
            info_dict = YT.extract_info(link, download=False)
            convert_video_to_audio_ffmpeg(right_title + '.mp4')
        return right_title + '.wav', info_dict['duration'], chapters
    except Exception as e:
        print('Не удалось скачать видео по ссылке')
        print(e)

def check_title(title):
    title = title.replace(' ', '_')
    title = re.sub(r'[^\w]', '_', title)
    title = re.sub(r'_{2,}', '_', title)
    title = re.sub(r'_$', '', title)

    return title

def convert_video_to_audio_ffmpeg(video_file, output_ext="wav"):
    """Converts video to audio directly using `ffmpeg` command
    with the help of subprocess module"""
    filename, ext = os.path.splitext(video_file)

    subprocess.call(["ffmpeg", "-y", "-i", video_file, f"{filename}.{output_ext}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)


def split_video(file_path: str, start: int, end: int, output_path: str):
    start_time = f'{start // 60}:{start % 60}:00'
    end_time = f'{end // 60}:{end % 60}:00'
    if start == end:
        end_time = f'{end // 60}:{end % 60}:30'
    command = f"ffmpeg -i {file_path} -ss {start_time} -to {end_time} -c copy {output_path}"
    subprocess.call(command, shell=True)
