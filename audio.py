import re

from yt_dlp import YoutubeDL
import subprocess

import math
import os


class AudioFile:
    """    """
    def __init__(self, s, duration):
        self.filename = s
        self.ext = self.filename[s.index('.'):]
        self.folder_name = self.filename[:s.index('.')] + '\\'
        self.abs_filename = f'C:\\Users\\dondu\\OneDrive\\Documents\\GitHub\\dyplom\\{self.filename}'
        self.duration = duration
        self.create_folder()

    def create_folder(self):
        print(f"Создание директории для файла {self.filename}")
        # создание папки для хранения исходного файла и его производных
        current_dir = os.path.dirname(os.path.realpath(__file__))
        dest_folder = os.path.join(current_dir, self.folder_name)
        if not os.path.isdir(dest_folder):
            os.mkdir(dest_folder)
        elif not (os.path.isfile(dest_folder + self.filename)):
            # перемещение исходного файла и его аудио в созданную папку
            # TODO : проверить удаляется ли файл
            os.replace(self.abs_filename, dest_folder + self.filename)
            self.abs_filename = dest_folder + self.filename

    def convert_and_split(self, abs_filename):
        """abs_filename"""
        splits = []

        total_mins = math.ceil(self.duration/ 60)
        #folder_name = self.abs_filename[:abs_filename.index('.')] + '\\'
        counter = 0
        try:
            if (10 < total_mins):
                    for i in range(0, total_mins, 5):
                        counter += 1
                        out = f'.\\{self.folder_name}\\{counter}_{self.filename}'
                        split_video(abs_filename, i, i + 5, out)
                        convert_video_to_audio_ffmpeg(out[:out.rindex('.')], "wav")
                        splits.append(out)

            else:
                out = f'.\\{self.folder_name}\\{self.filename}'
                convert_video_to_audio_ffmpeg(out[:out.rindex('.')], "wav")
                splits.append(out)
        except Exception as e:
                print(e)
        else:
            print("Конвертирование и разделение прошло успешно")
            return splits

def download_audio(link):
    if link.__contains__('youtube'):
        with (YoutubeDL()) as ydl:
            info_dict = ydl.extract_info(link, download=False)
            title = info_dict['title']
            right_title = check_title(title)

        with (YoutubeDL({'extract_audio': True, 'format': 'bestaudio/best',
                        'postprocessors':[{
                           'key':'FFmpegExtractAudio',
                            'preferredcodec':'wav',
                            'preferredquality':'192',
                        }],
                        'outtmpl': right_title,
                        }) as YT):

            #информация о видео по ссылке
            info_dict = YT.extract_info(link, download = False)
        return right_title +'.wav', info_dict['duration']


def check_title(title):
    title = title.replace(' ', '_')
    title = re.sub(r'[^\w]' , '_', title)
    title = re.sub(r'_{2,}' , '_', title)
    title = re.sub(r'_$' , '', title)

    return title

def get_length(input_video):
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
         input_video], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return float(result.stdout.strip())
def convert_video_to_audio_ffmpeg(video_file, output_ext="mp3"):
    """Converts video to audio directly using `ffmpeg` command
    with the help of subprocess module"""
    filename, ext = os.path.splitext(video_file)


    subprocess.call(["ffmpeg", "-y", "-i", video_file, f"{filename}.{output_ext}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)

def split_video(file_path, start, end, output_path):
    start_time = f'{start//60}:{start % 60 }:00'
    end_time = f'{end//60}:{end % 60}:00'
    command = f"ffmpeg -i {file_path} -ss {start_time} -to {end_time} -c copy {output_path}"
    subprocess.call(command, shell=True)





