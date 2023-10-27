import os
import pandas as pd
import joblib
from functions import *

# Esse código lê os resultados gerados durante o treinamento de cada modelo, gera um DataFrame com todos eles
# e depois exporta para uma planilha

# Encontrando a pasta e os arquivos .pkl com os resultados
pasta_resultados = "resultados_pkl"
lista_de_arquivos = os.listdir(pasta_resultados)
pkl_files = [file for file in lista_de_arquivos if file.endswith(".pkl")]

lista_resultados = []
nomes_resultados = []

# Carregando os arquivos .pkl
print("Nomes dos arquivos encontrados:\n")
for pkl_file in pkl_files:
    print(pkl_file)
    arquivo = joblib.load(os.path.join(pasta_resultados, pkl_file))

    if pkl_file.startswith('score_'):
        lista_resultados.append(arquivo)
        nomes_resultados.append(pkl_file)

print("\nLista de arquivos com scores: ")
print(nomes_resultados)
print("\nTamanho da lista de resultados: ", len(lista_resultados))
print("Tamanho da lista de nomes de resultados: ", len(nomes_resultados))

# Criando o DataFrame que irá conter os resultados
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

# Lendo os arquivos com os resultados e incluindo eles no DataFrame
for index, resultado in enumerate(lista_resultados):
    print("\nImprimindo resultados do arquivo ", nomes_resultados[index])

    # Identificando os atributos do modelo através do nome dele
    atributos_titulo = indentify_title(nomes_resultados[index])

    # if baseDados=='base1':
    #     baseDados='Train100'
    #     tamanhoBase='100000'
    # elif baseDados=='base2':
    #     baseDados='Train200'
    #     tamanhoBase='200000'
    # elif baseDados=='base3':
    #     baseDados='Train300'
    #     tamanhoBase='300000'
    # else: 
    #     tamanhoBase='Não identificado'

    # if tipoPreProcessamento=='doc2vec':
    #     tipoPreProcessamento='Limpo'
    #     vetorizacao='Doc2Vec'
    # elif tipoPreProcessamento=='limpo':
    #     tipoPreProcessamento='Limpo'
    #     vetorizacao='TF-IDF'
    # elif tipoPreProcessamento=='removeStopwords':
    #     tipoPreProcessamento='Remoção de Stopwords'
    #     vetorizacao='TF-IDF'
    # elif tipoPreProcessamento=='lem':
    #     tipoPreProcessamento='Lematização'
    #     vetorizacao='TF-IDF'
    # elif tipoPreProcessamento=='stem':
    #     tipoPreProcessamento='Stemming'
    #     vetorizacao='TF-IDF'
    # else:
    #     tipoPreProcessamento='Nenhum'
    #     vetorizacao='Nenhum'
   
    # if tipoAlgoritmo=='naiveBayes':
    #     tipoAlgoritmo='Naive Bayes'
    # elif tipoAlgoritmo=='regLog':
    #     tipoAlgoritmo='Logistic Regression'
    # elif tipoAlgoritmo=='svm':
    #     tipoAlgoritmo='Support Vector Machines'
    # elif tipoAlgoritmo=='extraTrees':
    #     tipoAlgoritmo='Extra Trees Classifier'
    # elif tipoAlgoritmo=='randomForest':
    #     tipoAlgoritmo='Random Forest'
    # elif tipoAlgoritmo=='random_predictions':
    #     tipoAlgoritmo='Teste Aleatório'
    # else:
    #     tipoAlgoritmo='Não identificado'

    if atributos_titulo[4] != 'Teste Aleatório':
        f1_score_mean_result = resultado['test_f1'].mean()
        balanced_accuracy_result = resultado['test_balanced_accuracy'].mean()
        roc_auc_score_result = resultado['test_roc_auc'].mean()
        precision_result = resultado['test_precision'].mean()
        recall_result = resultado['test_recall'].mean()
    else: 
        print("RESULTADO DO TESTE ALEATORIO: ", resultado)
        f1_score_mean_result = resultado[0]
        balanced_accuracy_result = resultado[1]
        roc_auc_score_result = resultado[2]
        precision_result = resultado[3]
        recall_result = resultado[4]

    # Exibindo as médias
    print("\nExibindo as médias do teste:")
    print("F1 Score:", f1_score_mean_result)
    print("Balanced Accuracy:", balanced_accuracy_result)
    print("ROC AUC Score:", roc_auc_score_result)
    print("Precision:", precision_result)
    print("Recall:", recall_result)
    
    # Incluindo os resultados desse modelo no DataFrame
    nova_linha = {
        'Base de dados': atributos_titulo[0], 
        'Tamanho da base de dados (em linhas)': atributos_titulo[1], 
        'Pré-processamento': atributos_titulo[2], 
        'Vetorização': atributos_titulo[3], 
        'Algoritmo': atributos_titulo[4],
        'F1 Score': f1_score_mean_result,
        'Acurácia Balanceada': balanced_accuracy_result,
        'ROC AUC Score': roc_auc_score_result,
        'Precisão': precision_result,
        'Recall': recall_result,
    }
    planilha_resultados = pd.concat([planilha_resultados, pd.DataFrame([nova_linha])], ignore_index=True)

# Exportando o DataFrame
planilha_resultados.to_excel('resultados_dos_testes.xlsx')