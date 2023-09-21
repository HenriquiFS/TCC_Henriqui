import os
import pandas as pd
import numpy as np
import random
import re
import nltk
import gensim
import joblib
import xgboost as xgb
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize
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
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from functions import *

start_time = datetime.now()

# Importando as bases de dados
base_1 = pd.read_csv('Train100.csv', sep=';')
base_2 = pd.read_csv('Train200.csv', sep=';')
base_3 = pd.read_csv('Train300.csv', sep=';')
bases_de_dados = [base_1, base_2, base_3]

doc2vec_model_file = Doc2Vec.load("doc2vec_models/meu_modelo_doc2vec.pt")

# lista_algoritmos = ['regLog', 'svm', 'extraTrees', 'randomForest', 'naiveBayes']
lista_algoritmos = ['svm']
# pre_processamentos = ['limpo', 'removeStopwords', 'lem', 'stem', 'doc2vec']
pre_processamentos = ['limpo', 'doc2vec']
metricas = ['f1', 'balanced_accuracy', 'roc_auc', 'precision', 'recall']
tfidf_vectorizer = TfidfVectorizer()
pasta_resultados = 'resultados'

lista_emoticons = {
    ':))': 'positive_emoticon', 
    ':)': 'positive_emoticon', 
    ':D': 'positive_emoticon', 
    ':(': 'negative_emoticon', 
    ':((': 'negative_emoticon', 
    '8)': 'neutral_emoticon'}
lista_abrev = {
    'eh': 'é',
    'vc': 'você',
    'vcs': 'vocês',
    'tb': 'também',
    'tbm': 'também',
    'obg': 'obrigado',
    'gnt': 'gente',
    'q': 'que',
    'n': 'não',
    'cmg': 'comigo',
    'p': 'para',
    'ta': 'está',
    'to': 'estou',
    'vdd': 'verdade',
    'ent': 'então',
    'ngm': 'ninguém',
    'tão': 'estão',
    'hj': 'hoje',
    'pq': 'porque',
    'tbem': 'também',
}
lista_stopwords = [
    'que',
    '...',
    '«',
    '➔',
    '|',
    '»',
    'positive_emoticon',
    'negative_emoticon',
    '#',
    '.',
    ',',
    '-',
    ':',
    'e',
    ';',
    '(',
    ',',
    'ma',
    'na',
    'no',
    'a',
    'à',
    'o',
    'em',
    'ou',
    'num',
    '"',
    'de',
    'do',
    'da',
    '[',
    ']',
]

for i, base in enumerate(bases_de_dados):
    comentarios, classes = return_columns(base, 'tweet_text', 'sentiment')
    base_index = str(i+1)
    print("\nRealizando treinamentos a base de dados ", base_index)

    for item in pre_processamentos:
        for algoritmo in lista_algoritmos:
            alg = choose_algorithm(algoritmo)
            if(item=='limpo'):
                print("Executando com pré processamento limpo")

                pipeline = Pipeline([
                    ('vectorizer', tfidf_vectorizer),
                    ('classifier', alg)
                ])

                pipeline.fit(comentarios, classes)
                pipeline_title = 'modelo_' + 'base' + base_index + '_' + algoritmo + '_' + item + '.pkl'
                destino = os.path.join(pasta_resultados, pipeline_title)
                export_to_pkl(pipeline, destino)

                scores_limpo = cross_validate(pipeline, comentarios, classes, cv=10, scoring=metricas)
                scores_title = 'score_' + 'base' + base_index + '_' + algoritmo + '_' + item + '.pkl'
                destino = os.path.join(pasta_resultados, scores_title)
                export_to_pkl(scores_limpo, destino)

                # Exibindo as médias
                # print("\nExibindo as médias do teste:\n")
                # print("F1 Score:", scores_limpo['test_f1'].mean())
                # print("Balanced Accuracy:", scores_limpo['test_balanced_accuracy'].mean())
                # print("ROC AUC Score:", scores_limpo['test_roc_auc'].mean())
                # print("Precision:", scores_limpo['test_precision'].mean())
                # print("Recall:", scores_limpo['test_recall'].mean())

            elif(item=='removeStopwords'):
                print("Executando com a remoção de stop words")

                comentarios_formatados = comentarios
                comentarios_formatados = format_data(
                    column=comentarios_formatados, 
                    emoti_list=lista_emoticons, 
                    abrev_list=lista_abrev, 
                    stopwords_list=lista_stopwords)
                comentarios_formatados = untokenize_text(comentarios_formatados)
                
                pipeline = Pipeline([
                    ('vectorizer', tfidf_vectorizer),
                    ('classifier', alg)
                ])

                pipeline.fit(comentarios_formatados, classes)
                pipeline_title = 'modelo_' + 'base' + base_index + '_' + algoritmo + '_' + item + '.pkl'
                destino = os.path.join(pasta_resultados, pipeline_title)
                export_to_pkl(pipeline, destino)

                scores = cross_validate(pipeline, comentarios_formatados, classes, cv=10, scoring=metricas)
                scores_title = 'score_' + 'base' + base_index + '_' + algoritmo + '_' + item + '.pkl'
                destino = os.path.join(pasta_resultados, scores_title)
                export_to_pkl(scores, destino)

            elif(item=='lem'):
                print("Executando com a Lematização")

                comentarios_formatados = comentarios
                comentarios_formatados = format_data(
                    column=comentarios_formatados, 
                    emoti_list=lista_emoticons, 
                    abrev_list=lista_abrev, 
                    stopwords_list=lista_stopwords)
                comentarios_formatados = apply_lemmatizer(comentarios_formatados)
                comentarios_formatados = untokenize_text(comentarios_formatados)
                
                pipeline = Pipeline([
                    ('vectorizer', tfidf_vectorizer),
                    ('classifier', alg)
                ])

                pipeline.fit(comentarios_formatados, classes)
                pipeline_title = 'modelo_' + 'base' + base_index + '_' + algoritmo + '_' + item + '.pkl'
                destino = os.path.join(pasta_resultados, pipeline_title)
                export_to_pkl(pipeline, destino)

                scores = cross_validate(pipeline, comentarios_formatados, classes, cv=10, scoring=metricas)
                scores_title = 'score_' + 'base' + base_index + '_' + algoritmo + '_' + item + '.pkl'
                destino = os.path.join(pasta_resultados, scores_title)
                export_to_pkl(scores, destino)

            elif(item=='stem'):
                print("Executando com Stemming")

                comentarios_formatados = comentarios
                comentarios_formatados = format_data(
                    column=comentarios_formatados, 
                    emoti_list=lista_emoticons, 
                    abrev_list=lista_abrev, 
                    stopwords_list=lista_stopwords)
                comentarios_formatados = apply_stemmer(comentarios_formatados)
                comentarios_formatados = untokenize_text(comentarios_formatados)
                
                pipeline = Pipeline([
                    ('vectorizer', tfidf_vectorizer),
                    ('classifier', alg)
                ])

                pipeline.fit(comentarios_formatados, classes)
                pipeline_title = 'modelo_' + 'base' + base_index + '_' + algoritmo + '_' + item + '.pkl'
                destino = os.path.join(pasta_resultados, pipeline_title)
                export_to_pkl(pipeline, destino)

                scores = cross_validate(pipeline, comentarios_formatados, classes, cv=10, scoring=metricas)
                scores_title = 'score_' + 'base' + base_index + '_' + algoritmo + '_' + item + '.pkl'
                destino = os.path.join(pasta_resultados, scores_title)
                export_to_pkl(scores, destino)

            elif(item=='doc2vec'):
                print("Executando com Doc2Vec")

                comentarios_formatados = comentarios
                comentarios_formatados = format_data(
                    column=comentarios_formatados, 
                    emoti_list=lista_emoticons, 
                    abrev_list=lista_abrev, 
                    stopwords_list=lista_stopwords)
                comentarios_formatados = untokenize_text(comentarios_formatados)

                pipeline = Pipeline([
                    ('vectorizer', Doc2VecVectorizer(doc2vec_model_file)),
                    ('classifier', alg)
                ])

                pipeline.fit(comentarios_formatados, classes)
                model_title = 'modelo_' + 'base' + base_index + '_' + algoritmo + '_' + item + '.pkl'
                destino = os.path.join(pasta_resultados, model_title)
                export_to_pkl(pipeline, destino)

                scores_limpo = cross_validate(pipeline, comentarios_formatados, classes, cv=10, scoring=metricas)
                scores_title = 'score_' + 'base' + base_index + '_' + algoritmo + '_' + item + '.pkl'
                destino = os.path.join(pasta_resultados, scores_title)
                export_to_pkl(scores_limpo, destino)   

            else:
                print("Não foi possível identificar um método de pré processamento")

end_time = datetime.now()
print("\nTempo de execução do começo até aqui: {}".format(end_time - start_time))