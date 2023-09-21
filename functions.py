import os
import pandas as pd
import numpy as np
import random
import re
import nltk
import gensim
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
from nltk.metrics import ConfusionMatrix
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Essa função recebe um data frame e retorna duas colunas
def return_columns(dataFrame, a_col, b_col):
    
    # print("Imprimindo uma amostra do Data Frame:\n", dataFrame.sample(5))

    comentarios = dataFrame.loc[:, a_col].values
    classes = dataFrame.loc[:, b_col].values

    return comentarios, classes

# Definindo uma função para remover URLs
def remove_url(data):
    ls = []
    words = ''
    regexp1 = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    regexp2 = re.compile('www?.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    for line in data:
        urls = regexp1.findall(line)

        for u in urls:
            line = line.replace(u, 'URL')

        urls = regexp2.findall(line)

        for u in urls:
            line = line.replace(u, 'URL')

        ls.append(line)
    return ls

# Definindo uma função para remover caracteres especificados
def remove_regex(data, regex_pattern):
    ls = []
    words = ''

    for line in data:
        matches = re.finditer(regex_pattern, line)

        for m in matches:
            line = re.sub(m.group().strip(), '', line)

        ls.append(line)

    return ls

# Definindo uma função para remover emoticons
def replace_emoticons(data, emoticon_list):
    ls = []

    for line in data:
        for exp in emoticon_list:
            line = line.replace(exp, emoticon_list[exp])

        ls.append(line)

    return ls

# Tokenização
def tokenize_text(data):
    ls = []

    for line in data:
        tokens = wordpunct_tokenize(line)
        ls.append(tokens)

    return ls

# Função para substituir tokens abreviados pelas palavras inteiras
def apply_standardization(tokens, std_list):
    ls = []

    for tk_line in tokens:
        new_tokens = []

        for word in tk_line:
            if word.lower() in std_list:
                word = std_list[word.lower()]

            new_tokens.append(word)

        ls.append(new_tokens)

    return ls

# Removendo stopwords
def remove_stopwords(tokens, stopword_list):
    ls = []

    for tk_line in tokens:
        new_tokens = []

        for word in tk_line:
            if word.lower() not in stopword_list:
                new_tokens.append(word)

        ls.append(new_tokens)

    return ls

# Aplicando Stemming
def apply_stemmer(tokens):
    nltk.download('rslp')
    ls = []
    stemmer = nltk.stem.RSLPStemmer()

    for tk_line in tokens:
        new_tokens = []

        for word in tk_line:
            word = str(stemmer.stem(word))
            new_tokens.append(word)

        ls.append(new_tokens)

    return ls

#Aplicando Lematização
def apply_lemmatizer(tokens):
    nltk.download('wordnet')
    lem = nltk.stem.WordNetLemmatizer()

    lem_tokens = []
    for tk_line in tokens:
        new_tokens = []
        for word in tk_line:
            new_word = lem.lemmatize(word)
            new_tokens.append(new_word)
        lem_tokens.append(new_tokens)

    return lem_tokens

# Distribuição de frequencia
def get_freq_dist_list(tokens):
    ls = []

    for tk_line in tokens:
        for word in tk_line:
            ls.append(word)

    return ls

# Transforma os tokens em frases novamente
def untokenize_text(tokens):
    ls = []

    for tk_line in tokens:
        new_line = ''

        for word in tk_line:
            new_line += word + ' '

        ls.append(new_line)

    return ls

# Realiza formatação dos dados até a remoção de stopwords
def format_data(column, emoti_list, abrev_list, stopwords_list):
    column = remove_url(column)
    column = remove_regex(column, '#[\w]*')
    column = remove_regex(column, '@[\w]*')
    column = replace_emoticons(column, emoti_list)
    column = tokenize_text(column)
    column = apply_standardization(column, abrev_list)
    column = remove_stopwords(column, stopwords_list)
    return column

# Escolhe e retorna um algoritmo
def choose_algorithm(alg):
        #Regressão Logísitica
        if (alg=='regLog'):
            print("\nExecutando o algoritmo de Regresão Logística")
            alg = LogisticRegression(max_iter=1000)

        # Support Vector Machines
        elif(alg=='svm'):
            print("\nExecutando o algoritmo de SVM")
            alg = svm.SVC()

        # Extra Trees Classifier
        elif(alg=='extraTrees'):
            print("\nExecutando o algoritmo de Extra Trees")
            alg = ExtraTreesClassifier(n_estimators=100, random_state=42)

        # Random Forest
        elif(alg=='randomForest'):
            print("\nExecutando o algoritmo de Random Forest")
            alg = RandomForestClassifier(n_estimators=100, random_state=42)

        # Naive Bayes
        elif(alg=='naiveBayes'):
            print("\nExecutando o algoritmo de Naive Bayes")
            alg = MultinomialNB()

        else:
            print("Não foi possível identificar um algoritmo")

        return alg

# Exportando o modelo
def export_to_pkl(model, title):
    joblib.dump(model, title)

# Identifica as palavras no título
def indentify_title(sentence):
    dataSet = ''
    algorithm = ''
    pre_processing = ''

    dataSet_list = [
        'base1',
        'base2',
        'base3'
    ]
    
    algorithms_list = [
        'regLog', 
        'svm', 
        'extraTrees', 
        'randomForest', 
        'naiveBayes'
    ]
    
    pre_processing_list = [ 
        'limpo', 
        'removeStopwords', 
        'lem', 
        'stem', 
        'doc2vec'
    ]

    for item in dataSet_list:
        if re.search(item, sentence):
            dataSet=item
    for item in algorithms_list:
        if re.search(item, sentence):
            algorithm=item
    for item in pre_processing_list:
        if re.search(item, sentence):
            pre_processing=item

    return dataSet, algorithm, pre_processing

class Doc2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, doc2vec_model):
        self.doc2vec_model = doc2vec_model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Transforma os documentos em vetores usando o modelo Doc2Vec
        return [self.doc2vec_model.infer_vector(doc.split()) for doc in X]