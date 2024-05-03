from yt_dlp import YoutubeDL
import subprocess

import math
import os
from pydub import AudioSegment


class AudioFile:
    """    """
    def __init__(self, s):
        self.filename = s
        self.ext = self.filename[s.index('.'):]
        self.folder_name = self.filename[:s.index('.')] + '\\'
        self.abs_filename = f'C:\\Users\\dondu\\OneDrive\\Documents\\GitHub\\dyplom\\{self.filename}'
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
        length = get_length(abs_filename)
        total_mins = math.ceil(length / 60)
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




class SplitWavAudioMubin():
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename
        self.filepath = folder + '\\' + filename

        self.audio = AudioSegment.from_wav(self.filepath)

    def get_duration(self):
        return self.audio.duration_seconds

    def single_split(self, from_min, to_min, split_filename):
        t1 = from_min * 60 * 1000
        t2 = to_min * 60 * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export(self.folder + '\\' + split_filename, format="wav")

    def multiple_split(self, min_per_split):
        total_mins = math.ceil(self.get_duration() / 60)
        list_splits = []
        for i in range(0, total_mins, min_per_split):
            split_fn = str(i) + '_' + self.filename
            self.single_split(i, i + min_per_split, split_fn)
            split_fn = self.folder + '\\' + str(i) + '_' + self.filename
            list_splits.append(split_fn)
            print(str(i) + ' Done')
            if i == total_mins - min_per_split:
                print('All splited successfully')
                return list_splits

def download_audio(link):
    if link.__contains__('youtube'):
        with (YoutubeDL({'extract_audio': True, 'format': 'bestaudio/best',
                        'postprocessors':[{
                           'key':'FFmpegExtractAudio',
                            'preferredcodec':'wav',
                            'preferredquality':'192',
                        }],
                        'outtmpl': '%(title)s',
                        }) as YT):
        #    YT.download(link)

            #информация о видео по ссылке
            info_dict = YT.extract_info(link, download = False)

            print(info_dict['title'])
            s = info_dict['title'].replace('.', '')
            s = s.replace('?', '')
            s = s.replace('|', '')
            s = s.replace('  ', '_')
            s = s.replace(' ', '_')
            print(s)
            os.rename(info_dict['title']+'.wav', s+'.wav')
            return s + '.wav'


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





