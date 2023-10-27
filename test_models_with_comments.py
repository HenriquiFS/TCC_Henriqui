import os
import pandas as pd
import numpy as np
import random
import re
import pickle
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from functions import *

# O objetivo desse código é testar a performance da cada modelo ao classificar uma base de dados personalizada

# Definindo as métricas que serão usadas na hora de calcular a pontuação
cv_number = 10
metricas = ['f1', 'balanced_accuracy', 'roc_auc', 'precision', 'recall']

# Lendo a base de dados
comentarios_de_teste = pd.read_csv('comentarios_gerados.csv', sep=';')
comentarios, classes = return_columns(comentarios_de_teste, 'tweet_text', 'sentiment')

# Encontrando os modelos
pasta_resultados = "resultados_pkl"
lista_de_arquivos = os.listdir(pasta_resultados)
pkl_files = [file for file in lista_de_arquivos if file.endswith(".pkl")]


lista_modelos = []
nomes_modelos = []

# Carregando os modelos
print("Nomes dos arquivos encontrados:\n")
for pkl_file in pkl_files:
    print(pkl_file)
    arquivo = joblib.load(os.path.join(pasta_resultados, pkl_file))

    if pkl_file.startswith('modelo_'):
        lista_modelos.append(arquivo)
        nomes_modelos.append(pkl_file)

# Criando um DataFrame que será usado para exportar os resultados
df_resultados = pd.DataFrame(columns=[
    'Modelo', 
    'Base de dados do modelo', 
    'Pré-processamento', 
    'Vetorização', 
    'Algoritmo',
    'Acurácia Balanceada', 
    'F1-score', 
    'Precisão',
    'Recall',
    'ROC AUC Score'
])

# Usando cada modelo para classificar os comentários da base de dados personalizada
for index, modelo in enumerate(lista_modelos):
    print("\nUtilizando o modelo ", nomes_modelos[index], " para realizar previsões")

    atributos_titulo = indentify_title(nomes_modelos[index])
    print("\nImprimindo atributos do título: ", atributos_titulo)

    scores = cross_validate(modelo, comentarios, classes, cv=cv_number, scoring=metricas)
    # Exibindo as médias
    print("\nExibindo as médias do teste:\n")
    print("F1 Score:", scores['test_f1'].mean())
    print("Balanced Accuracy:", scores['test_balanced_accuracy'].mean())
    print("ROC AUC Score:", scores['test_roc_auc'].mean())
    print("Precision:", scores['test_precision'].mean())
    print("Recall:", scores['test_recall'].mean())

    # Adicionando uma nova linha no DataFrame com os resultados do teste
    nova_linha = {
        'Modelo': nomes_modelos[index], 
        'Base de dados do modelo': atributos_titulo[0], 
        'Pré-processamento': atributos_titulo[2], 
        'Vetorização': atributos_titulo[3], 
        'Algoritmo': atributos_titulo[4],
        'Acurácia Balanceada': scores['test_balanced_accuracy'].mean(), 
        'F1-score': scores['test_f1'].mean(), 
        'Precisão': scores['test_precision'].mean(),
        'Recall': scores['test_recall'].mean(),
        'ROC AUC Score': scores['test_roc_auc'].mean()
    }
    df_resultados = pd.concat([df_resultados, pd.DataFrame([nova_linha])], ignore_index=True)

# Exportando o DataFrame
df_resultados.to_excel('resultados_com_base_comentarios.xlsx')