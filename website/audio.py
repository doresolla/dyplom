import re
from time import time

import requests
from pytubefix import YouTube
import subprocess

from pywhispercpp.model import Model
import os


media_root = "/home/ubuntu/dyplom/media/videos"
class AudioFile:
    def __init__(self, s, chapters, isThere,video_id, abs=''):
        self.filename = os.path.basename(s)
        self.folder_name = self.filename[:self.filename.rindex('.')]
        if abs == '':
            path_to_file = f"/home/ubuntu/dyplom/media/videos/{self.folder_name}/" + self.filename
            if os.path.isfile(path_to_file):
                self.abs_filename = path_to_file
            else:
                path_to_file = f"/home/ubuntu/dyplom/media/videos/" + self.filename
                if os.path.isfile(path_to_file):
                    self.abs_filename = path_to_file
        else:
            self.abs_filename = abs
        self.chapters = chapters
        self.video_id = video_id
        self.abs_folder = ''
        self.model_cpp = Model('small', n_threads=2, language='ru', translate=False)
        if not isThere:
            self.create_folder()

    def create_folder(self):
        # создание папки для хранения исходного файла и его производных
        dest_folder = os.path.join(media_root, self.folder_name)
        self.abs_folder = dest_folder
        if not os.path.isdir(dest_folder):
            os.mkdir(dest_folder)
            print(f"Создание директории для файла {self.filename}")
        if not (os.path.isfile(os.path.join(dest_folder, self.filename))):
            # перемещение исходного файла и его аудио в созданную папку
            before = self.abs_filename
            after = os.path.join(dest_folder, self.filename)
            os.replace(before,  after)
            before = self.abs_filename[:self.abs_filename.rindex('.')] + '.wav'
            after = os.path.join(dest_folder, self.filename[:self.filename.rindex('.')] + '.wav')
            os.replace(before,after)
            before = os.path.join(media_root, f'{self.filename[:self.filename.rindex('.')]}_thumb.jpg')
            after = os.path.join(dest_folder, f'{self.filename[:self.filename.rindex('.')]}_thumb.jpg')

            os.replace(before,after)

            self.abs_filename = os.path.join(dest_folder, self.filename)
            self.abs_folder = dest_folder
            print(f"Перемещение  файла {self.filename} в директорию {self.folder_name}")


    def recognizePywhisper_cpp(self):
        txt_file = os.path.join(self.abs_folder, self.filename[:self.filename.rindex('.')] + '.txt')
        if (not os.path.isfile(txt_file)):
            # Если такого файла не существует
            start_time = time()
            print('Началось распознавание')
            # segments = model.transcribe(self.filename, speed_up=True)
            segments = self.model_cpp.transcribe(self.abs_filename)
            end_time = time()
            recognized_text = ''
            timings_file = os.path.join(self.abs_folder,  'timings.txt')
            with open(timings_file, 'w', encoding='utf-8') as f:
                for seg in segments:
                    recognized_text += seg.text + ' '
                    f.write(f'{seg.t0}\n{seg.text}\n{seg.t1}\n\n')
            print("Время выполнения, ", end_time - start_time)
            with open(os.path.join(self.abs_folder, txt_file), 'w', encoding='utf-8') as f:
                print(f'В файл {os.path.join(self.abs_folder, txt_file)} записан текст из аудио')
                f.write(recognized_text)
        else:
            print(f'Файл {txt_file} уже есть')


    def extract_image(self):
        width = 1920
        height = 1080
        filename, ext = os.path.splitext(self.abs_filename)
        print("Имя видео файла", filename + '.mp4')
        s = "eq(pict_type\,PICT_TYPE_I)"
        command = ["ffmpeg", "-y", "-i", filename + '.mp4', "-vsync", "0", "-vf", f"select={s}", f"-s",
                   f"{width}x{height}", "-f", "image2", f"{filename + '\\' + filename}-%03d.jpeg"]
        try:
            result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)
            print("Команда успешно выполнена")
        except subprocess.CalledProcessError as e:
            print(f"Ошибка выполнения команды: {e}")


def download(link: str):
    try:
        proxies={'http':'http://127.0.0.1:8881',
                 'https':'http://127.0.0.1:8881'}

        yt = YouTube(url=link, proxies=proxies)
        right_title = check_title(yt.title)

        # Скачиваем только аудио (mp4 контейнер с AAC)
        video_stream = yt.streams. \
            filter(type='video'). \
            order_by('resolution'). \
            desc().first()
        audio_stream = yt.streams. \
            filter(mime_type='audio/mp4'). \
            order_by('filesize'). \
            desc().first()
        if not audio_stream:
            print("Аудио-поток не найден")
            return None, 0, []
        print('Download video...')
        video_stream.download()
        print('\nDownload audio...')
        audio_stream.download()
        # wav_filename = f"{right_title}.wav"
        combine(audio_stream.default_filename, video_stream.default_filename,
                f'{yt.title}.mp4')
        # Конвертация в WAV через ffmpeg
        # subprocess.run([
        #     "ffmpeg", "-y",  # -y = overwrite
        #     "-i", mp4_filename,
        #     "-ac", "1",  # моно (можно убрать)
        #     "-ar", "16000",  # частота дискретизации 16kHz (можно изменить)
        #     wav_filename
        # ], check=True)

        return audio_stream.default_filename, []

    except Exception as e:
        print("Ошибка при загрузке и конвертации:")
        print(e)
        return None, 0, []

def combine(audio: str, video: str, output: str) -> None:
    if os.path.exists(output):
        os.remove(output)
    code = os.system(f'.\\ffmpeg.exe -i "{video}" -i "{audio}" -c copy "{output}"')
    if code != 0:
        pass
    raise SystemError(code)

def check_title(title):
    title = title.replace(' ', '_')
    title = re.sub(r'[^\w]', '_', title)
    title = re.sub(r'_{2,}', '_', title)
    title = re.sub(r'_$', '', title)
    title = title.strip()
    return title


def convert_video_to_audio_ffmpeg(video_file, output_ext="wav"):
    filename, ext = os.path.splitext(video_file)
    subprocess.call(["ffmpeg", "-y", "-i", video_file, f"{filename}.{output_ext}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)


def split_video(file_path: str, start: int, end: int, output_path: str):
    start_time = f'{start // 60}:{start % 60}:00'
    end_time = f'{end // 60}:{end % 60}:00'
    if start == end:
        end_time = f'{end // 60}:{end % 60}:30'
    command = f"ffmpeg -i {file_path} -ss {start_time} -to {end_time} -n -c copy {output_path}"
    subprocess.call(command, shell=True)


def get_length(input_video):
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
         input_video], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return float(result.stdout.strip())


