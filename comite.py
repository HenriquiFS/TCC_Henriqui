from functions import *
import pickle
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score, roc_auc_score

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# O objetivo desse código é juntar vários modelos e formar um comitê, em que cada algoritmo tem um voto na hora
# de classificar um comentário, por isso é importante escolher um número ímpar de algoritmos

# nomes_das_bases = ['comentarios_gerados.csv', 'Train100.csv', 'Train200.csv', 'Train300.csv']

# Escolhendo uma base de dados:
nome_da_base = 'comentarios_gerados.csv'
base_dados = pd.read_csv(nome_da_base, sep=';')
comentarios, classes = return_columns(base_dados, 'tweet_text', 'sentiment')

# Encontrando os arquivos .pkl que contém os modelos já treinados
pasta_resultados = "modelos_escolhidos"
lista_de_arquivos = os.listdir(pasta_resultados)
pkl_files = [file for file in lista_de_arquivos if file.endswith(".pkl")]

lista_modelos = []
nomes_modelos = []

# Lendo os arquivos
print("Nomes dos arquivos encontrados:")
for pkl_file in pkl_files:
    print(pkl_file)
    arquivo = joblib.load(os.path.join(pasta_resultados, pkl_file))

    lista_modelos.append(arquivo)
    nomes_modelos.append(pkl_file)

# Realizando previsões com cada modelo
pred1 = lista_modelos[0].predict(comentarios)
pred2 = lista_modelos[1].predict(comentarios)
pred3 = lista_modelos[2].predict(comentarios)
pred4 = lista_modelos[3].predict(comentarios)
pred5 = lista_modelos[4].predict(comentarios)
pred6 = lista_modelos[5].predict(comentarios)
pred7 = lista_modelos[6].predict(comentarios)

# Implementando o comitê
ensemble_predictions = np.vstack((pred1, pred2, pred3, pred4, pred5, pred6, pred7)).T
final_predictions = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=ensemble_predictions)

# Calculando as métricas
accuracy = accuracy_score(classes, final_predictions)
balanced_accuracy = balanced_accuracy_score(classes, final_predictions)
f1 = f1_score(classes, final_predictions)
precision = precision_score(classes, final_predictions)
recall = recall_score(classes, final_predictions)
roc_auc = roc_auc_score(classes, final_predictions)

print("Resultados com a base ", nome_da_base)
print("\nAcurácia:", accuracy)
print("Acurácia Balanceada:", balanced_accuracy)
print("F1-score:", f1)
print("Precisão:", precision)
print("Recall:", recall)
print("ROC AUC Score:", roc_auc)

print("\nGerando a matriz de confusão...")
matriz_confusao = confusion_matrix(classes, final_predictions)
sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Rótulos Previstos')
plt.ylabel('Rótulos Reais')
plt.show()
# plt.savefig('matriz_confusao.png', format='png') # Não funcionou corretamente
# np.save('matriz_confusao.png', matriz_confusao)

if nome_da_base=='comentarios_gerados.csv':

    # Criando um array com os resultados das previsões, que vai virar uma columa em um DataFrame
    string_predictions = []
    string_real = []
    for i in range(len(final_predictions)):
        if final_predictions[i] == 1:
            string_predictions.append('positivo')
        elif final_predictions[i] == 0:
            string_predictions.append('negativo')

    for i in range(len(classes)):
        if classes[i] == 1:
            string_real.append('positivo')
        elif classes[i] == 0:
            string_real.append('negativo')

    # Criando e exportando o DataFrame
    data_predictions = {'Comentários': comentarios, 'Real': classes, 'Previsões': string_predictions}
    df_predictions = pd.DataFrame(data_predictions)
    df_predictions.to_excel('teste_com_comentarios_gerados.xlsx')
