import subprocess
import os, re
from pytubefix import YouTube
from django.conf import settings
PROXIES={'http':'http://127.0.0.1:8881',
        'https':'http://127.0.0.1:8881'}

def read_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        message_to_user = "[Файл не найден или ошибка чтения]"
        return message_to_user

def check_title(title):
    title = title.replace(' ', '_')
    title = re.sub(r'[^\w]', '_', title)
    title = re.sub(r'_{2,}', '_', title)
    title = re.sub(r'_$', '', title)
    title = title.strip()
    return title

def get_video_duration(path):
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries',
             'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True
        )
        duration = float(result.stdout.strip())
        return int(duration), None
    except Exception as e:
        message_to_user = f'Ошибка: не удалось получить длительность видео. {e} '
        print(message_to_user)
        return 0, message_to_user


def extract_thumbnail(video_path, thumbnail_path):
    try:
        subprocess.run([
            'ffmpeg', '-i', video_path,
            '-ss', '00:00:01.000', '-vframes', '1',
            thumbnail_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, stdin=subprocess.DEVNULL)
        message_to_user = 'extract_thumbnail'
        print(message_to_user)
        return True, message_to_user
    except Exception as e:
        print('extract_thumbnail FALSE')
        message_to_user = f'extract_thumbnail FALSE {e}'
        return False, message_to_user

def download(link: str):
    try:
        yt = YouTube(url=link, proxies=PROXIES)
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
        video_stream.download(output_path=os.path.join(settings.MEDIA_ROOT, 'videos'))
        print('\nDownload audio...')
        audio_stream.download(output_path=os.path.join(settings.MEDIA_ROOT, 'videos'))
        # wav_filename = f"{right_title}.wav"
        combine(audio_stream.default_filename, video_stream.default_filename,
                f'{right_title}.mp4')
        duration = yt.length
        return audio_stream.default_filename, duration, [],'Скачано видео'

    except Exception as e:
        message_to_user = f"Ошибка при загрузке и конвертации:{e}"
        print(message_to_user)
        return None, 0, [], message_to_user

def combine(audio: str, video: str, output: str) -> None:
    if os.path.exists(output):
        os.remove(output)
    # code = os.system(f'.\\ffmpeg.exe -i "{video}" -i "{audio}" -c copy "{output}"')
    subprocess.call(["ffmpeg", "-y", "-i", video, '-i', audio, '-c copy', output],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)
    # if code != 0:
    #     pass
    # raise SystemError(code)
