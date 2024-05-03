from keras.layers import TextVectorization
import string, re
from tensorflow import strings
def tokenize(text):

    dict = text

    # Создаем экземпляр токенизатора
    tokenizer = TextVectorization(
        max_tokens=10,
        standardize='lower_and_strip_punctuation',
        output_mode='tf_idf',
        )

    tokenizer.adapt(dict)
    # Выведем список всех слов в словаре
    tokenizer.adapt(dict)


    word_indexes = [tokenizer(sentence) for sentence in dict]
    print(word_indexes)

    for i in range(len(tokenizer.get_vocabulary())):
        print(f'{i} -->', tokenizer.get_vocabulary()[i])

    for i in word_indexes:
        print("e.numpy() = ", i.numpy())

def lower_standardization(input_data):
  lowercase = strings.lower(input_data)
  return strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')