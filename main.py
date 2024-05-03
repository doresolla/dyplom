import text, audio, os
import whisper

class Main:
    def __init__(self, type):
        self.model = whisper.load_model(type)


def recognizeSpeech(audioname, model):
    filename = audioname[:audioname.rindex('\\')] + audioname[
                                                    audioname.rindex('_') + 1:audioname.rindex('.')] + '.txt'
    print('Start', audioname[audioname.rindex('\\') + 1:])
    result = model.transcribe(audioname)
    # model_base = whisper.load_model("large-v2")

    # файл с расшифровкой речи
    # Если такого файла не существует
    if not (os.path.isfile(filename)):
        param = 'w'  # создать файл и записать в него
    else:
        param = 'a+'  # добавить данные в конец файла

    file = open(filename, param)
    file.write(result['text'] + '\n')
    file.close()
    print("End", audioname[audioname.rindex('\\') + 1:])


if __name__ == '__main__':
    filename = audio.download_audio('https://www.youtube.com/watch?v=vPVAYsc-U60&list=PLz-Ep7rzAifojTH64KGZfJF1DL4KoNRp4&index=3')

    file = audio.AudioFile(filename)
    file.convert_and_split(file.abs_filename)
    files = audio.convert_video_to_audio_ffmpeg(f'.\\{file.folder_name}\\{file.filename}.{file.ext}', "wav")
    model = Main("base")

    for f in files:
        try:
            recognizeSpeech(f, model.model)
        except Exception as e:
            print(e)
        else:
            print("Recognition successfully Ended")

    text_to_tokenize = ["Это кот съел все конфеты!", "Кот также съел это мясо"]
    text_to_tokenize = [sen.lower() for sen in text_to_tokenize]
    eng = ['This cat ate all sweets!', 'The cat also ate this meat']
    text.tokenize(text_to_tokenize)


