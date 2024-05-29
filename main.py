import audio
from text import Text, read_subs
import whisper
import torch
from os import path
import nltk
nltk.download('punkt')

if __name__ == '__main__':
    # links =  ['https://www.youtube.com/watch?v=I_ReFF3qiQ8',
    #           # 'https://www.youtube.com/watch?v=jR6x5PmBL2I',
    #           # 'https://www.youtube.com/watch?v=6dYPBA7-1Wg'
    #       #    'https://www.youtube.com/watch?v=dgDp5T2-1ec',
    #           ]
    # for link in links:
    #     filename, duration, chapters = audio.download_audio(link)
    #     print(chapters)
    #     file = audio.AudioFile(filename, duration, chapters)
    #     files = file.split(file.abs_filename)
    #     print(files)
    #
    #     torch.cuda.is_available()
    #     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    #     model = whisper.load_model("small", DEVICE, in_memory=True )
    #     start = file.duration // 10 // 60
    #     current_dir = path.dirname(path.realpath(__file__))
    #     dest_folder = path.join(current_dir, file.folder_name)
    #     cut = dest_folder + 'cut.wav'
    #     audio.split_video(file.folder_name + file.filename , start, start, cut)
    #     cut = whisper.load_audio(cut)
    #     cut = whisper.pad_or_trim(cut)
    #     mel = whisper.log_mel_spectrogram(cut).to(model.device)
    #     _, probs = model.detect_language(mel)
    #
    #     output = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    #
    #     print("Вероятности появления различных языков:", output)
    #     lang = output[0]
    #     print("Язык - ", lang)
    #     if output[1][1] > 0.2:
    #         print(output[1][1])
    #     try:
    #         file.recognizeSpeech(files, model)
    #     except Exception as e:
    #         print('Ошибка при распознавании голоса')
    #         print(e)
    #     else:
    #         print("Распознавание прошло успешно")
        name = "Протокол_HTTP_Компьютерные_сети_2024_10"
      #  name = file.filename[:file.filename.index('.')]
    #ltmmatize
        text = Text(name)
        text.split_paragraphs()
        text.sent_summary()

