import os.path
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
from networkx import pagerank, from_numpy_array, Graph
from sklearn.feature_extraction.text import CountVectorizer
from pymorphy3 import MorphAnalyzer
from math import exp
from re import compile, sub

TOKEN = compile('\w+')
stopWords = set(stopwords.words("russian"))
class Text:
    def __init__(self, name, chapters=None):
        if chapters is None:
            chapters = []
        self.name = name
        with open(name + '\\' + name + '.txt', 'rb') as f:
            data = f.read(1000)
        result = detect(data)
        print(result['encoding'])
        recognized_text = ''
        with open(name + '\\' + name + '.txt', encoding='utf-8') as file_text:
            chunk = file_text.read(10000)
            while chunk:
                recognized_text = recognized_text + chunk
                chunk = file_text.read(10000)
        self.text = recognized_text

        REG = compile('\w+')
        text_no_punkt = " ".join(REG.findall(self.text))

        self.data = {"Исходный текст": recognized_text, "Без пунктуации": text_no_punkt}
        self.tokenize()

        self.chapters = chapters
        self.paragraphs = self.split_paragraphs()
        self.morph = MorphAnalyzer()

        self.model = load('ru_core_news_md', exclude=['parser', 'attribute_ruler', 'morphologizer'])

    def tokenize(self):
        sentences = sent_tokenize(self.text)
        self.sentences = [sent[:len(sent) - 1] for sent in sentences]
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

    def split_paragraphs(self):
        print('Разделение текста на параграфы')
        model = SentenceTransformer('DiTy/bi-encoder-russian-msmarco')
        sentence_length = [len(each) for each in self.sentences]
        # long = np.mean(sentence_length) + np.std(sentence_length) * 2
        # text = ''
        # for each in self.sentences:
        #     if len(each) > long:
        #         # let's replace all the commas with dots
        #         each = each.replace(',', '.')
        #     text += f'{each}. '
        # sentences = text.split('. ')
        # Embed sentences
        embeddings = model.encode(self.sentences)
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
        paragraphs = []
        para = ''
        count = 0
        for num, sen in enumerate(self.sentences):
            if num in split_points:
                count += 1
                paragraphs.append(para)
                para = f'{sen}. '
                text += f'\n {sen}. '
            else:
                text += f'{sen}. '
                para += f'{sen}. '
        print('paragraphs', len(paragraphs))
        # print('\n'.join(paragraphs))
        with open(self.name + "\\paragraphs.txt", 'w', encoding="utf-8") as f:
            f.write(text)
        return paragraphs

    def lemmatize(self, text):
        words = text.split()  # разбиваем текст на слова
        if len(words) > 1:
            res = list()
            for word in words:
                p = self.morph.parse(word.lower())[0]
                res.append(p.normal_form)
                return res
        else:
            p = self.morph.parse(words[0].lower())[0]
            return p.normal_form

    def sent_para_summary(self):
        if not self.tokens or not self.sentences:
            return
        with open(self.name + "\\" + "sent_para_summary.txt", 'a+', encoding='utf-8') as f:
            if not self.model.has_pipe('sentencizer'):
                self.model.add_pipe('sentencizer')
            doc = self.model(self.text)
            # sentences = [sent.text[:-1] for sent in doc.sents]
            sentences = sent_tokenize(self.text, language='russian')
            sentences = [sen[:-1] for sen in sentences]
            tokens = word_tokenize(" ".join(compile('\w+').findall(self.text)), language='russian')
            lemmas = self.lemmatizer_spacy(doc)
            self.data['Лемматизированный текст'] = " ".join(lemmas)
            freqTable = dict()
            for word in tokens:
                word = self.lemmatize(word)
                if word in stopWords:
                    continue
                if word in freqTable:
                    freqTable[word] += 1
                else:
                    freqTable[word] = 1
            sentenceValue = dict()
            allFreq = 0
            for sentence in sentences:
                for word, freq in freqTable.items():
                    if word.lower() in sentence.lower():
                        if sentence in sentenceValue:
                            sentenceValue[sentence] += freq
                        else:
                            sentenceValue[sentence] = freq
                try:
                    allFreq += sentenceValue[sentence]
                except Exception as e:
                    print(e)

            average = int(allFreq / len(sentenceValue))

            for para in self.paragraphs:
                sen_para = sent_tokenize(para, language='russian')
                sen_para = [sen[:-1] for sen in sen_para]
                for sentence in sen_para:
                    if sentence in sentenceValue and (sentenceValue[sentence] > (1.1 * average)):
                        f.write(sentence + ". ")
                f.write('\n\n')

    def sent_summary(self):
        if not self.tokens or not self.sentences:
            return
        doc = self.model(self.data['Без пунктуации'].lower())

        print('Именнованные сущности')
        ents = [ent for ent in doc.ents]
        print(ents, sep=', ')

        lemmas = self.lemmatizer_spacy(doc)
        self.data['Лемматизированный текст'] = " ".join(lemmas)
        freqTable = dict()
        for word in self.tokens:
            if word in self.stopWords:
                continue
            if word in freqTable:
                freqTable[word] += 1
            else:
                freqTable[word] = 1
        sentenceValue = dict()
        allFreq = 0
        for sentence in self.sentences:
            for word, freq in freqTable.items():
                if word.lower() in sentence.lower():
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
        with open(self.name + "\\" + "sent_summary.txt", 'w', encoding='utf-8') as f:
            for sentence in self.sentences:
                if (sentenceValue[sentence] > (1.1 * average)) or (any(str(element) in sentence for element in set(ents))):
                    f.write(sentence.strip() + ". ")



    def text_rank(self, num_sentences=5):
        sentences = []
        sentence_paragraph_map = []
        for para_idx, paragraph in enumerate(self.paragraphs):
            para_sentences = sent_tokenize(paragraph)
            sentences.extend(para_sentences)
            sentence_paragraph_map.extend([para_idx] * len(para_sentences))

        vectorizer = CountVectorizer().fit_transform(sentences)
        vectors = vectorizer.toarray()
        cosine_matrix = cosine_similarity(vectors)

        graph = from_numpy_array(cosine_matrix)
        scores = pagerank(graph)
        ranked_sentences = sorted(((score, idx) for idx, score in scores.items()), reverse=True)

        top_sentence_indices = [idx for score, idx in ranked_sentences[:num_sentences]]
        top_sentence_indices.sort()

        summary = [sentences[idx] for idx in top_sentence_indices]
        summary_t = ' '.join(summary)

        with open(self.name+'\\text_rank.txt', 'w', encoding='utf-8')as f:
            f.write(summary_t)
        return summary_t

    def add_data_export(self, dataset):

        if (os.path.isfile(dataset)):
            df = pd.read_csv(dataset, index_col=0)
            if self.data['Исходный текст'] not in df['Исходный текст'].values:
                df_copy = pd.DataFrame(self.data, index=[1])
                df = pd.concat([df_copy, df], ignore_index=True)
                if 'Unnamed: 0' in df.columns:
                    df.drop(columns=['Unnamed: 0'], inplace=True)
        else:
            df = pd.DataFrame(self.data, index=[0])
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
    new_text = sub(r'(\d{2}:\d{2}:\d{2}\.\d{3})|(<c>)|(</c>)|(align:start position:0%)|(-->)|(\s{2})', '', recognized_text)
    new_text = sub(r'(\s{3})|(<>)', '', new_text)
    new_text = new_text.replace('WEBVTT '
                                'Kind: captions'
                                'Language: ru', "")

    with open(new_file, 'w') as new_file:
        new_file.write(new_text)
        print('Сохранено')


def build_similarity_matrix(sentences):
    vectorizer = CountVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix


def pagerank_sentences(sentences, similarity_matrix):
    nx_graph = from_numpy_array(similarity_matrix)
    scores = pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    return ranked_sentences


def rev_sigmoid(x: float) -> float:
    return (1 / (1 + exp(0.5 * x)))


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


def sentence_similarity(sent1, sent2):
    words1 = set(sent1)
    words2 = set(sent2)

    common_words = words1.intersection(words2)
    if not common_words:
        return 0
    return len(common_words) / (len(words1) + len(words2))

def build_graph(sentences):
    graph = Graph()
    for i, sent1 in enumerate(sentences):
        for j, sent2 in enumerate(sentences):
            if i != j:
                similarity = sentence_similarity(sent1, sent2)
                if similarity > 0:
                    graph.add_edge(i, j, weight=similarity)
    return graph

def rank_sentences(graph):
    scores = pagerank(graph, weight='weight')
    ranked_sentences = sorted(((score, idx) for idx, score in scores.items()), reverse=True)
    return ranked_sentences

def tokenize_and_lemmatize(sentence):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(sentence)
    lemmas = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalnum() and
              token.lower() not in stopWords]
    return lemmas

