import text, audio, os
import whisper
import chardet
import torch
from string import punctuation

if __name__ == '__main__':
    link = 'https://www.youtube.com/watch?v=6dYPBA7-1Wg&t=2s'
    is_here = False
    if not (is_here):
        filename, duration, chapters = audio.download_audio(link, True)
        file = audio.AudioFile(filename, duration, chapters)
        files = file.convert_and_split(file.abs_filename)
        print(files)

        torch.cuda.is_available()
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("base", DEVICE, in_memory=True )
        try:
            file.recognizeSpeech(files, model)
        except Exception as e:
            print('Ошибка при распознавании голоса')
            print(e)
        else:
            print("Распознавание прошло успешно")
    else:
        name = 'СШГЭС_4_Авария.wav'
        file_audio = audio.AudioFile(name, 843, [] )
        name = file_audio.folder_name + '\\' + file_audio.filename.replace(file_audio.ext, '.txt')
        print(name)
        with open(name, 'rb') as f:
            data = f.read(1000)
            f.seek(0, os.SEEK_END)
            bytes = f.tell()
            print("Размер текстового файла", bytes)

        # Detect the encoding of the data
        result = chardet.detect(data)
        print(result['encoding'])
        recognized_text = ''
        with open(name, encoding=result['encoding']) as file_text:
            chunk = file_text.read(10000)
            while chunk:
                recognized_text = recognized_text + chunk
                chunk = file_text.read(10000)
        print(recognized_text)
        text.read_subs(file_audio.folder_name)
        # prep = text.preprocess()
        # sents, tokens = prep.sent_vector(recognized_text)
        # print(sents, end = '\n')
        # print(tokens, end = ' ')

