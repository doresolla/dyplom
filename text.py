import os.path

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, sent_tokenize
import pandas as pd
from spacy import load
from chardet import detect
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import argrelextrema

import re, math

TOKEN = re.compile('\w+')
stopWords = set(stopwords.words("russian"))
model = load('ru_core_news_sm', exclude=['parser', 'attribute_ruler', 'ner','morphologizer'])
#morph = MorphAnalyzer()

class Text:
    def __init__(self, name):
        self.name = name
        with open(name + '\\' + name+'.txt', 'rb') as f:
            data = f.read(1000)
        result = detect(data)
        print(result['encoding'])
        recognized_text = ''
        with open(name + '\\'+ name+'.txt', encoding='utf-8') as file_text:
            chunk = file_text.read(10000)
            while chunk:
                recognized_text = recognized_text + chunk
                chunk = file_text.read(10000)
        self.text = recognized_text
        REG = re.compile('\w+')
        text_no_punkt = " ".join(REG.findall(self.text))
        self.data = {"Исходный текст":recognized_text,"Без пунктуации":text_no_punkt}
        self.tokenize()


    def tokenize(self):
        sentences = sent_tokenize(self.text)
        self.sentences = [sent[:len(sent)-1] for sent in sentences]
        tokens = word_tokenize(self.data['Без пунктуации'].lower(), language='russian')
        tokens = [token for token in tokens if token not in stopWords]
        self.tokens = list(dict.fromkeys(tokens))
        print("Количество предложений исходного текста:", len(self.sentences))


    def lemmatizer_spacy(self, doc):
        return [token.lemma_ for token in doc]
    def stemm(self, text):
        stemmer = SnowballStemmer(language="russian")
        stem = []
        for token in text:
            stem.append(stemmer.stem(token))
        stems = " ".join(stem)
        self.data['stem'] = stems
        return stem

    def sent_summary(self):
        if not self.tokens or not self.sentences:
            return
        doc = model(self.data['Без пунктуации'].lower())
        lemmas = self.lemmatizer_spacy(doc)
        self.data['Лемматизированный текст'] = " ".join(lemmas)
        freqTable = dict()
        for word in self.tokens:
            if word in stopWords:
                continue
            if word in freqTable:
                freqTable[word] += 1
            else:
                freqTable[word] = 1
        sentenceValue = dict()
        allFreq = 0
        for sentence in self.sentences:
            for word, freq in freqTable.items():
                if word in sentence.lower():
                    if sentence in sentenceValue:
                        sentenceValue[sentence] += freq
                    else:
                        sentenceValue[sentence] = freq
            try:
                allFreq += sentenceValue[sentence]
            except Exception as e:
                print(e)

        average = int(allFreq / len(sentenceValue))

        # Storing sentences into our summary.
        with open(self.name +"\\"+"sent_summary.txt", 'a+', encoding='utf-8') as f:
            for sentence in self.sentences:
                if (sentenceValue[sentence] > (1.1 * average)):
                    sentence = sentence.strip()
                    f.write(sentence + ". ")

    def split_paragraphs(self):
        print('Разделение текста на параграфы')
        model = SentenceTransformer('DiTy/bi-encoder-russian-msmarco')
        # Get the length of each sentence
        sentece_length = [len(each) for each in self.sentences]
        # Determine longest outlier
        long = np.mean(sentece_length) + np.std(sentece_length) * 2
        # Determine shortest outlier
        short = np.mean(sentece_length) - np.std(sentece_length) * 2
        # Shorten long sentences
        text = ''
        for each in self.sentences:
            if len(each) > long:
                # let's replace all the commas with dots
                each = each.replace(',', '.')
            text += f'{each}. '
        sentences = text.split('. ')
        # Embed sentences
        embeddings = model.encode(sentences)
        print(embeddings.shape)

        # Normalize the embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        # Create similarities matrix
        similarities = cosine_similarity(embeddings)
        activated_similarities = activate_similarities(similarities, p_size=10)

        minmimas = argrelextrema(activated_similarities, np.less, order=2)
        # Create empty string
        split_points = [each for each in minmimas[0]]
        text = ''
        for num, each in enumerate(sentences):
            if num in split_points:
                text += f'\n\n {each.capitalize()}. '
            else:
                text += f'{each}. '
        with open(self.name + "\\paragraphs.txt", 'w',  encoding="utf-8") as f:
            f.write(text)

    def add_data_export(self, dataset):
        if (os.path.isfile(dataset)):
            df = pd.read_csv(dataset, index_col=0)
            if self.data['Исходный текст'] not in df['Исходный текст'].values:
                df_copy = pd.DataFrame(self.data, index=[1])
                df = pd.concat([df_copy, df], ignore_index=True)
                if 'Unnamed: 0' in df.columns:
                    df.drop(columns=['Unnamed: 0'], inplace=True)
        else:
            df = pd.DataFrame(self.data, index = [0])
        df.to_csv(dataset)

def read_subs(folder):
    name = folder + '\\' + 'sub.srt.ru.vtt'
    recognized_text = ''
    with open(name) as f:
        chunk = f.read(10000)
        while chunk:
            recognized_text = recognized_text + chunk
            chunk = f.read(10000)
    new_file = folder + '\\' + 'new_sub.srt.ru.vtt'
    new_text = re.sub(r'(\d{2}:\d{2}:\d{2}\.\d{3})|(<c>)|(</c>)|(align:start position:0%)|(-->)|(\s{2})', '',
                      recognized_text)
    new_text = re.sub(r'(\s{3})|(<>)', '', new_text)
    new_text = new_text.replace('WEBVTT '
                                'Kind: captions'
                                'Language: ru', "")

    with open(new_file, 'w') as new_file:
        new_file.write(new_text)
        print('сохранено')

def rev_sigmoid(x: float) -> float:
    return (1 / (1 + math.exp(0.5 * x)))

def activate_similarities(similarities: np.array, p_size=10) -> np.array:
    """ Function returns list of weighted sums of activated sentence similarities
    Args:
        similarities (numpy array): it should square matrix where each sentence corresponds to another with cosine similarity
        p_size (int): number of sentences are used to calculate weighted sum
    Returns:
        list: list of weighted sums
    """
    # To create weights for sigmoid function we first have to create space. P_size will determine number of sentences used and the size of weights vector.
    x = np.linspace(-10, 10, p_size)
    # Then we need to apply activation function to the created space
    y = np.vectorize(rev_sigmoid)
    # Because we only apply activation to p_size number of sentences we have to add zeros to neglect the effect of every additional sentence and to match the length ofvector we will multiply
    activation_weights = np.pad(y(x), (0, similarities.shape[0] - p_size))
    ### 1. Take each diagonal to the right of the main diagonal
    diagonals = [similarities.diagonal(each) for each in range(0, similarities.shape[0])]
    ### 2. Pad each diagonal by zeros at the end. Because each diagonal is different length we should pad it with zeros at the end
    diagonals = [np.pad(each, (0, similarities.shape[0] - len(each))) for each in diagonals]
    ### 3. Stack those diagonals into new matrix
    diagonals = np.stack(diagonals)
    ### 4. Apply activation weights to each row. Multiply similarities with our activation.
    diagonals = diagonals * activation_weights.reshape(-1, 1)
    ### 5. Calculate the weighted sum of activated similarities
    activated_similarities = np.sum(diagonals, axis=0)
    return activated_similarities