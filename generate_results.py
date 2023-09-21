import os
import pandas as pd
import joblib
from functions import *

pasta_resultados = "resultados"
lista_de_arquivos = os.listdir(pasta_resultados)
pkl_files = [file for file in lista_de_arquivos if file.endswith(".pkl")]

# Carrega os modelos a partir dos arquivos .pkl
lista_modelos = []
nomes_modelos = []
lista_resultados = []
nomes_resultados = []

print("Nomes dos arquivos encontrados:\n")
for pkl_file in pkl_files:
    print(pkl_file)
    arquivo = joblib.load(os.path.join(pasta_resultados, pkl_file))

    if pkl_file.startswith('modelo_'):
        lista_modelos.append(arquivo)
        nomes_modelos.append(pkl_file)
    elif pkl_file.startswith('score_'):
        lista_resultados.append(arquivo)
        nomes_resultados.append(pkl_file)
    else:
        print("Não foi possível identificar o tipo do arquivo")

print("\nLista de arquivos com models: ")
print(nomes_modelos)
print("\nLista de arquivos com scores: ")
print(nomes_resultados)

print("tamanho da lista de resultados: ", len(lista_resultados))
print("tamanho da lista de nomes de resultados: ", len(nomes_resultados))

planilha_resultados = pd.DataFrame(columns=[
    'Base de dados', 
    'Tamanho da base de dados (em linhas)',
    'Pré-processamento', 
    'Vetorização', 
    'Algoritmo',
    'F1 Score',
    'Acurácia Balanceada',
    'ROC AUC Score',
    'Precisão',
    'Recall',
])

for index, resultado in enumerate(lista_resultados):
    print("\nImprimindo resultados do arquivo ", nomes_resultados[index])
    # print(resultado)

    baseDados, tipoAlgoritmo, tipoPreProcessamento = indentify_title(nomes_resultados[index])

    if baseDados=='base1':
        baseDados='Train100'
        tamanhoBase='100000'
    elif baseDados=='base2':
        baseDados='Train200'
        tamanhoBase='200000'
    elif baseDados=='base3':
        baseDados='Train300'
        tamanhoBase='300000'
    else: 
        tamanhoBase='Não identificado'

    if tipoPreProcessamento=='doc2vec':
        tipoPreProcessamento='Limpo'
        vetorizacao='Doc2Vec'
    elif tipoPreProcessamento=='removeStopwords':
        tipoPreProcessamento='Remoção de Stopwords'
        vetorizacao='TF-IDF'
    elif tipoPreProcessamento=='lem':
        tipoPreProcessamento='Lematização'
        vetorizacao='TF-IDF'
    elif tipoPreProcessamento=='stem':
        tipoPreProcessamento='Stemming'
        vetorizacao='TF-IDF'
    else:
        tipoPreProcessamento='Limpo'
        vetorizacao='TF-IDF'
   
    if tipoAlgoritmo=='naiveBayes':
        tipoAlgoritmo='Naive Bayes'
    elif tipoAlgoritmo=='regLog':
        tipoAlgoritmo='Logistic Regression'
    elif tipoAlgoritmo=='svm':
        tipoAlgoritmo='Support Vector Machines'
    elif tipoAlgoritmo=='extraTrees':
        tipoAlgoritmo='Extra Trees Classifier'
    elif tipoAlgoritmo=='randomForest':
        tipoAlgoritmo='Random Forest'
    else:
        tipoAlgoritmo='Não identificado'

    f1_score_mean = resultado['test_f1'].mean()
    balanced_accuracy = resultado['test_balanced_accuracy'].mean()
    roc_auc_score = resultado['test_roc_auc'].mean()
    precision = resultado['test_precision'].mean()
    recall = resultado['test_recall'].mean()

    # Exibindo as médias
    print("\nExibindo as médias do teste:")
    print("F1 Score:", f1_score_mean)
    print("Balanced Accuracy:", balanced_accuracy)
    print("ROC AUC Score:", roc_auc_score)
    print("Precision:", precision)
    print("Recall:", recall)
    
    nova_linha = {
        'Base de dados': baseDados, 
        'Tamanho da base de dados (em linhas)': tamanhoBase, 
        'Pré-processamento': tipoPreProcessamento, 
        'Vetorização': vetorizacao, 
        'Algoritmo': tipoAlgoritmo,
        'F1 Score': f1_score_mean,
        'Acurácia Balanceada': balanced_accuracy,
        'ROC AUC Score': roc_auc_score,
        'Precisão': precision,
        'Recall': recall,
    }

    planilha_resultados = pd.concat([planilha_resultados, pd.DataFrame([nova_linha])], ignore_index=True)

planilha_resultados.to_excel('resultados_dos_testes.xlsx')

# Agora a lista loaded_models contém todos os modelos carregados a partir dos arquivos .pkl na pasta especificada.
# resultados = joblib.load('scores1_teste.pkl')
# print("Imprimindo o conteúdo do pickle: ", resultados)