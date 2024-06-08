from audio import AudioFile
from text import Text, read_subs
import whisper
from os import path

if __name__ == '__main__':
    links =  ['https://www.youtube.com/watch?v=I_ReFF3qiQ8',
              #'https://www.youtube.com/watch?v=jR6x5PmBL2I',
            #  'https://www.youtube.com/watch?v=6dYPBA7-1Wg',
     #   'https://www.youtube.com/watch?v=ML5tP8m6SHw',
     #    'https://www.youtube.com/watch?v=k9 wK2FThEsk&t'
              ]
    for link in links:
        # filename, duration, chapters = audio.download_audio(link, False)
        # print(chapters)
        # file = audio.AudioFile(filename, duration, chapters)
        filename = 'Протокол_HTTP_Компьютерные_сети_2024_10.wav'
        # #files = file.split(file.abs_filename)
        # files = ['1_Протокол_HTTP_Компьютерные_сети_2024_10.wav',
        #  '2_Протокол_HTTP_Компьютерные_сети_2024_10.wav',
        #  '3_Протокол_HTTP_Компьютерные_сети_2024_10.wav',
        #  '4_Протокол_HTTP_Компьютерные_сети_2024_10.wav'
        #  ]
      #  print(files)
      #   start = file.duration // 10 // 60
      #   current_dir = path.dirname(path.realpath(__file__))
      #   dest_folder = path.join(current_dir, file.folder_name)
      #   cut = dest_folder + 'cut.wav'
      #   audio.split_video(file.folder_name + file.filename , start, start, cut)
      #   model = whisper.load_model("small", in_memory=True)
      #
      #   cut = whisper.load_audio(cut)
      #   cut = whisper.pad_or_trim(cut)
      #   mel = whisper.log_mel_spectrogram(cut).to(model.device)
      #   _, probs = model.detect_language(mel)
      #
      #   output = sorted(probs.items(), key=lambda x: x[1], reverse=True)
      #   print("Вероятности появления различных языков:", output)
      #   lang = output[0]
      #   print("Язык - ", lang)
      #   #если есть неопределенность в языке
      #   if output[1][1] > 0.2:
      #       print(output[1])
        try:
          #  file.recognizeSpeech([file.folder_name + file.filename])
            file = AudioFile(filename, 1070, [])
            file.recognizePywhisper_cpp()
        except Exception as e:
            print('Ошибка при распознавании голоса')
            print(e)
        else:
            print("Распознавание прошло успешно")
    #     name = file.filename[:file.filename.index('.')]
    #    # name = 'Протокол_HTTP_Компьютерные_сети_2024_10'
    #  #   name = 'СШГЭС_4_Авария'
    #    # name = '_В_ЕГЭ_по_математике_нет_сложных_задач_Задания_1_11_Профильный_уровень_Борис_Трушин'
    # #ltmmatize
    #     text = Text(name)
    #     text.split_paragraphs()
    #     text.sent_summary()
    #     text.add_data_export('dataset.csv')
    #
