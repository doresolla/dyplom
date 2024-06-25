import re
from time import time
from yt_dlp import YoutubeDL
import subprocess

from pywhispercpp.model import Model
import os


class AudioFile:

    def __init__(self, s, duration, chapters, isThere, abs=''):
        self.filename = s
        self.folder_name = self.filename[:s.index('.')] + '\\'
        if abs == '':
            self.abs_filename = os.path.dirname(os.path.realpath(__file__)) + '\\' + self.filename
        else:
            self.abs_filename = abs
        self.duration = duration
        self.chapters = chapters
        self.model_cpp = Model('small', n_threads=6, language='ru')
        if not isThere:
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
            os.replace(self.abs_filename[:self.abs_filename.index('.')] + '.mp4',
                       dest_folder + self.filename[:self.filename.index('.')] + '.mp4')
            try:
                os.replace(current_dir + '\\tmp\\sub.srt.ru.vtt', dest_folder + '\\sub.srt.ru.vtt')
            except Exception as e:
                print('Нет субтитров', e)
            try:
                os.replace(current_dir + '\\chapters.txt', dest_folder + '\\chapters.txt')
            except Exception as e:
                print('Нет глав', e)
            self.abs_filename = dest_folder + self.filename
            print(f"Перемещение  файла {self.filename} в директорию {self.folder_name}")

    # def split(self, abs_filename):
    #     splits = []
    #     total_mins = ceil(self.duration / 60)
    #     counter = 0
    #     try:
    #         if (10 < total_mins):
    #             for i in range(0, total_mins, 5):
    #                 counter += 1
    #                 out = f'.\\{self.folder_name}{counter}_{self.filename}'
    #                 split_video(abs_filename, i, i + 5, out)
    #                 splits.append(out)
    #
    #         else:
    #             out = f'.\\{self.folder_name}\\{self.filename}'
    #             splits.append(out)
    #     except Exception as e:
    #         print("Ошибка при разделении")
    #         print(e)
    #     else:
    #         print("Разделение прошло успешно")
    #         return splits

    # def recognizeSpeech(self, files):
    #     # файл с расшифровкой речи
    #     txt_file = self.filename[:self.filename.rindex('.')] + '.txt'
    #     start_time = time()
    #     for f in files:
    #         # текущий файл
    #         print('Start', f[f.rindex('\\') + 1:])
    #         result = self.model.transcribe(f, fp16=False, language='ru')
    #         if not (os.path.isfile(self.folder_name + txt_file)):
    #             param = 'w'  # создать файл и записать в него
    #         else:
    #             param = 'a+'  # добавить данные в конец файла
    #         with open(self.folder_name+ txt_file, param,  encoding="utf-8") as file:
    #             file.write(result['text'] + ' ')
    #         print("Конец", f[f.rindex('\\') + 1:])
    #     print("Время выполнения, ", time() - start_time)

    def recognizePywhisper_cpp(self):
        txt_file = self.filename[:self.filename.rindex('.')] + '.txt'
        # Если такого файла не существует
        try:
            start_time = time()

            # segments = model.transcribe(self.filename, speed_up=True)
            segments = self.model_cpp.transcribe(self.folder_name + self.filename)
            end_time = time()
            recognized_text = ''
            with open(self.folder_name + 'timings.txt', 'w', encoding='utf-8') as f:
                for seg in segments:
                    recognized_text += seg.text + ' '
                    f.write(f'{seg.t0}\n{seg.text}\n{seg.t1}\n\n')
            print("Время выполнения, ", end_time - start_time)
            with open(self.folder_name + txt_file, 'w', encoding='utf-8') as f:
                print(self.folder_name + txt_file)
                f.write(recognized_text)

            # for f in files:
            #     print(f)
            #     result = w.transcribe_from_file(f)
            #     print(result)
            #     print('Конец', f)
            # if not (os.path.isfile(self.folder_name + txt_file)):
            #     param = 'w'  # создать файл и записать в него
            # else:
            #     param = 'a+'  # добавить данные в конец файла
            # with open(self.folder_name+ txt_file, param,  encoding="utf-8") as file:
            #     file.write(result['text'] + ' ')
        except Exception as e:
            print(e)

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


def download_audio(link: str):
    try:
        chapters = []
        with (YoutubeDL({
            'writeautomaticsub': True,
            'subtitlesformat': 'srt',
            'skip_download': True,
            'subtitleslangs': ['ru'],
            'outtmpl': '/tmp/sub.srt'
        })) as ydl:
            # if to_download:
            #     ydl.download(link)
            info_dict = ydl.extract_info(link, download=False)
            # разделение на главы (end_time, start_time, title)

        right_title = check_title(info_dict['title'])
        if ('chapters' in info_dict):
            chapters = info_dict['chapters']
            with open('chapters.txt', 'w', encoding='utf-8') as f:
                print(chapters)
                for chapter in chapters:
                    f.write(f'{chapter['start_time']} \t{chapter['title']}\t{chapter['end_time']}\n')

            with (YoutubeDL({
                'format': 'best[ext=mp4][height<=720]',
                'outtmpl': '{}.%(ext)s'.format(right_title),  # Имя файла будет основано на названии видео
                'merge_output_format': 'mp4',  # Формат выходного файла
                'no_warnings':True,
                'concurrent-fragments': 6
            }) as YT):
                dir = os.path.dirname(os.path.realpath(__file__))
            if (not os.path.exists(dir + right_title)):
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
