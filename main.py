from yt_dlp import YoutubeDL
import subprocess
import wave
import whisper
import math
import os
from pydub import AudioSegment


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
         #   YT.download(link)

            #информация о видео по ссылке
            info_dict = YT.extract_info(link, download = False)

            return info_dict['title'] + '.wav'

def recognizeSpeech(audioname):
    filename = audioname[:audioname.rindex('\\')] + audioname[audioname.rindex('_') + 1:audioname.rindex('.')] + '.txt'
    print('Start', audioname[audioname.rindex('\\') + 1 :])
    model_base = whisper.load_model("base")
    result_base = model_base.transcribe(audioname)
    # model_base = whisper.load_model("large-v2")
    #файл с расшифровкой речи
    #Если такого файла не существует
    if not (os.path.isfile(filename)):
        param = 'w' # создать файл и записать в него
    else:
        param = 'a+' #добавить данные в конец файла

    file = open(filename, param)
    file.write(result_base['text'] + '\n')
    file.close()
    print("End", audioname[audioname.rindex('\\') + 1 :])

def convert_video_to_audio_ffmpeg(video_file, output_ext="mp3"):
    """Converts video to audio directly using `ffmpeg` command
    with the help of subprocess module"""
    filename, ext = os.path.splitext(video_file)
    subprocess.call(["ffmpeg", "-y", "-i", video_file, f"{filename}.{output_ext}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)


if __name__ == '__main__':
    #filename = download_audio('')
    abs_filename = 'C:\\Users\\dondu\\Downloads\\stepic2.mp4'
    filename = abs_filename[abs_filename.rindex('\\')+1:]
    print("FILENAME = ", filename)
    #создание папки для хранения исходного файла и его производных
    folder_name = filename[:filename.index('.')] + '\\'
    current_dir = os.path.dirname(os.path.realpath(__file__))
    dest_folder = os.path.join(current_dir, folder_name)
    if not os.path.isdir(dest_folder):
        os.mkdir(dest_folder)
    elif not (os.path.isfile(dest_folder + filename)):
        #перемещение исходного файла и его аудио в созданную папку
        os.replace(abs_filename, dest_folder+filename)
    convert_video_to_audio_ffmpeg(f'.\\{folder_name}\\'+ filename, "wav")

    filename = filename[:filename.rindex('.')]
    print(filename)
    split_audio = SplitWavAudioMubin(dest_folder, filename + ".wav")
    files = split_audio.multiple_split( min_per_split=1)
    for f in files:
        recognizeSpeech(f)
    print("Recognition successfully Ended")