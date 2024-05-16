import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
import pandas as pd
from spacy import load
from chardet import detect
from spacy.lang.ru.examples import sentences
import numpy as np
# Library to import pre-trained model for sentence embeddings
from sentence_transformers import SentenceTransformer
# Calculate similarities between sentences
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import argrelextrema
from spacy.lang.ru import Russian
import re, math

TOKEN = re.compile('[\w\d]+')
class Text:
    def __init__(self, folder, name):
        #self.stopwords_ru = stopwords.words('russian')
        self.folder = folder
        self.name = name
        with open(self.folder + self.name, 'rb') as f:
            data = f.read(1000)
        result = detect(data)
        print(result['encoding'])
        recognized_text = ''
        with open(folder + name, encoding=result['encoding']) as file_text:
            chunk = file_text.read(10000)
            while chunk:
                recognized_text = recognized_text + chunk
                chunk = file_text.read(10000)
        self.text = recognized_text
        self.model = load('ru_core_news_sm')
        self.sentences = []
        self.tokens = []
        self.dataset = pd.DataFrame()

    def tokenize_regex(self, text):
        text = text.lower()
        all_tokens =TOKEN.findall(text)
        return all_tokens
       # return [token for token in all_tokens if token not in self.stopwords_ru]
    def tokenize_corpus(self, corpus, tokenizer=tokenize_regex, **tokenizer_kwargs):
        return [tokenizer(text, **tokenizer_kwargs) for text in corpus]

    def sent_vector(self):
        sentences = sent_tokenize(self.text)
        self.sentences = [sent[:len(sent)-1] for sent in sentences]
        for s in self.sentences:
            self.tokens.extend( self.tokenize_regex(s))

        return self.sentences, self.tokens
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

    def sent_summary(self):
        stopWords = set(stopwords.words("russian"))
        #words = word_tokenize(self.text)
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
            allFreq += sentenceValue[sentence]

        average = int(allFreq / len(sentenceValue))

        # Storing sentences into our summary.
        with open(self.folder+"sent_summary.txt", 'w' ) as f:
            for sentence in sentences:
                if (sentenceValue[sentence] > (1.2 * average)):
                    f.write(sentence + " ")
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
        with open(self.folder+"paragraphs.txt", 'w') as f:
            f.write(text)


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
        print('сохранено')

def rev_sigmoid(x:float)->float:
    return (1 / (1 + math.exp(0.5*x)))

def activate_similarities(similarities:np.array, p_size=10)->np.array:
        """ Function returns list of weighted sums of activated sentence similarities
        Args:
            similarities (numpy array): it should square matrix where each sentence corresponds to another with cosine similarity
            p_size (int): number of sentences are used to calculate weighted sum
        Returns:
            list: list of weighted sums
        """
        # To create weights for sigmoid function we first have to create space. P_size will determine number of sentences used and the size of weights vector.
        x = np.linspace(-10,10,p_size)
        # Then we need to apply activation function to the created space
        y = np.vectorize(rev_sigmoid)
        # Because we only apply activation to p_size number of sentences we have to add zeros to neglect the effect of every additional sentence and to match the length ofvector we will multiply
        activation_weights = np.pad(y(x),(0,similarities.shape[0]-p_size))
        ### 1. Take each diagonal to the right of the main diagonal
        diagonals = [similarities.diagonal(each) for each in range(0,similarities.shape[0])]
        ### 2. Pad each diagonal by zeros at the end. Because each diagonal is different length we should pad it with zeros at the end
        diagonals = [np.pad(each, (0,similarities.shape[0]-len(each))) for each in diagonals]
        ### 3. Stack those diagonals into new matrix
        diagonals = np.stack(diagonals)
        ### 4. Apply activation weights to each row. Multiply similarities with our activation.
        diagonals = diagonals * activation_weights.reshape(-1,1)
        ### 5. Calculate the weighted sum of activated similarities
        activated_similarities = np.sum(diagonals, axis=0)
        return activated_similarities
