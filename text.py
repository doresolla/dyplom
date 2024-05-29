import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize, sent_tokenize
import pandas as pd
from spacy import load
from spacy.lang.ru import Russian
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
        self.text = recognized_text.lower()
        REG = re.compile('\w+')
        text_no_punkt = " ".join(REG.findall(self.text))
        data = {"Исходный текст":self.text,"Без пунктуации":[text_no_punkt]}
        self.sentences, self.tokens = self.tokenize()
        self.dataset = pd.DataFrame(data)

    def tokenize(self):
        sentences = sent_tokenize(self.text)
        tokens = []
        self.sentences = [sent[:len(sent)-1] for sent in sentences]
        for s in self.sentences:
            tokens.extend(word_tokenize(s,language='russian'))
        tokens = [token for token in tokens if token not in stopWords]
        self.tokens = list(dict.fromkeys(tokens))
        return self.sentences, self.tokens

    def lemma_spacy(self):
        for doc in model.pipe(self.dataset['Без пунктуации'].values):
            lemma = [n.lemma_ for n in doc]
            self.dataset['lemma_ru_core'] = [lemma]
        print("Lemmatize is done")
    def lemmatizer_spacy(self, doc):
        lemmatizer = model.get_pipe("lemmatizer")
        print(lemmatizer.mode)  # какой лемматизатор используется
        lemmas = " ".join([token.lemma_ for token in doc])
        self.dataset['lemma_lemmatizer_spacy'] = [lemmas]
        return lemmas

    # def lemmatize(self, doc):
    #     words = []
    #     for token in doc:
    #         if (token.is_stop != True) and (token.is_punct != True) and (token.is_space != True) and (token.is_digit != True):
    #             words.append(token.lemma_)
    #     return ' '.join(words)
    def stemm(self):
        stemmer = SnowballStemmer(language="russian")
        stem = []
        for token in self.tokens:
            stem.append(stemmer.stem(token))
        stems = " ".join(stem)
        self.dataset['stem'] = stems
        return stems



    def sent_summary(self):
        if not self.tokens or not self.sentences:
            return
        docs = list(model.pipe(self.dataset['Исходный текст']))
        lemmas = self.lemmatizer_spacy(docs[0])
        self.dataset['Лемматизированный текст']=[lemmas]
        freqTable = dict()
        for word in lemmas:
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
            allFreq += sentenceValue[sentence]

        average = int(allFreq / len(sentenceValue))

        # Storing sentences into our summary.
        with open(self.name +"\\"+ self.name+"_sent_summary.txt", 'a+' ) as f:
            for sentence in self.sentences:
                if (sentenceValue[sentence] > (1.05 * average)):
                    sentence = sentence.strip()
                    f.write(sentence.capitalize() + ". ")
                    print(sentence)

    def split_paragraphs(self):
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
        # Now let's concatenate short ones
        text = ''
        for each in sentences:
            text += f'{each}. '
        # Split text into sentences
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
                text += f'\n\n {each}. '
            else:
                text += f'{each}. '
        with open(self.name + "\\paragraphs.txt", 'w') as f:
            f.write(text)

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