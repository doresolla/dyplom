import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
import pandas as pd
from spacy import load
from spacy.lang.ru.examples import sentences
from spacy.lang.ru import Russian
import string, re

TOKEN = re.compile('[\w\d]+')
class preprocess:
    def __init__(self):
        self.stopwords_ru = stopwords.words('russian')
        print(self.stopwords_ru)

     # нужно ли
        #   nlp = Russian()
        self.model = load('ru_core_news_sm')
        self.sentences = []
        self.dataset = pd.DataFrame()

    def tokenize_regex(self, text):
        text = text.lower()
        all_tokens = TOKEN.findall(text)
        return [token for token in all_tokens if token not in self.stopwords_ru]
    def tokenize_corpus(self, corpus, tokenizer=tokenize_regex, **tokenizer_kwargs):
        return [tokenizer(text, **tokenizer_kwargs) for text in corpus]

    def sent_vector(self, text):
        sentences = sent_tokenize(text)
        self.sentences = [sent[:len(sent)-1] for sent in sentences]
        tokens = [self.tokenize_regex(s) for s in self.sentences]
        return self.sentences, tokens
    def tokenize(self, text):


        # Создаем экземпляр токенизатора
        tokens = word_tokenize(text,language='ru', preserve_line=True)

        # Выведем список всех слов в словаре
        print(tokens)

        #
        # word_indexes = [tokenizer(sentence) for sentence in dict]
        # print(word_indexes)
        #
        # for i in range(len(tokenizer.get_vocabulary())):
        #     print(f'{i} -->', tokenizer.get_vocabulary()[i])
        #
        # for i in word_indexes:
        #     print("e.numpy() = ", i.numpy())

    def lemmatize(self):
        lemma = []
        #
        # for doc in self.model.pipe(author_data["text_clean"].values):
        #     lemma.append([n.lemma_ for n in doc])

def read_subs(folder):
    name = folder + '\\' + 'sub.srt.ru.vtt'
    recognized_text = ''
    with open(name) as f:
        chunk = f.read(10000)
        while chunk:
            recognized_text = recognized_text + chunk
            chunk = f.read(10000)
    new_file = folder+'\\'+'new_sub.srt.ru.vtt'
    new_text = re.sub(r'(\d{2}:\d{2}:\d{2}\.\d{3})|(<c>)|(</c>)|(align:start position:0%)|(-->)|(\s{2})', '', recognized_text)
    new_text = re.sub(r'(\s{3})|(<>)', '', new_text)
    new_text = new_text.replace('WEBVTT '
                                'Kind: captions'
                                'Language: ru', "")

    with open(new_file, 'w') as new_file:
        new_file.write(new_text)


