import audio
from text import Text, read_subs
import whisper
import torch


if __name__ == '__main__':
    link = 'https://www.youtube.com/watch?v=6dYPBA7-1Wg'
    is_here = True
    if not (is_here):
        filename, duration, chapters = audio.download_audio(link, False)
        file = audio.AudioFile(filename, duration, chapters)
        files = file.convert_and_split(file.abs_filename)
        print(files)

        torch.cuda.is_available()
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("small", DEVICE, in_memory=True )
        try:
            file.recognizeSpeech(files, model)
        except Exception as e:
            print('Ошибка при распознавании голоса')
            print(e)
        else:
            print("Распознавание прошло успешно")
    else:
        name = 'СШГЭС_4_Авария.txt'
        folder = 'СШГЭС_4_Авария\\'
        print(name)
        text = Text(folder, name)

        read_subs(folder)
        sents, tokens = text.sent_vector()
        text.split_paragraphs()

