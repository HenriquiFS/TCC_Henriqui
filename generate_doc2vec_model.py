import os
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import multiprocessing

print("Iniciando o procedimento...")

# Pasta contendo os documentos de texto
corpus_folder = 'documentos_doc2vec'

# Obtém a lista de nomes de arquivos na pasta
file_names = [os.path.join(corpus_folder, filename) for filename in os.listdir(corpus_folder)]

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

# Função para pré-processar o texto (remover stopwords, tokenizar, etc.)
def preprocess_text(text):
    # stop_words = set(stopwords.words('portuguese'))
    tokens = simple_preprocess(text, deacc=True)  # deacc=True remove acentos
    return [token for token in tokens if token not in lista_stopwords]

# Lista para armazenar documentos pré-processados
tagged_data = []

# Lê e pré-processa os documentos
for i, file_name in enumerate(file_names):
    print("Lendo o arquivo ", file_name)
    with open(file_name, 'r', encoding='utf-8') as file:
        try:
            text = file.read()
            preprocessed_text = preprocess_text(text)
            tagged_data.append(TaggedDocument(words=preprocessed_text, tags=[i]))
        except:
            print("Não foi possível utilizar o documento ", file_name)

# Configurações para treinamento do modelo
cores = multiprocessing.cpu_count()
vector_size = 100  # Tamanho dos vetores de documento
window_size = 10  # Tamanho da janela de contexto
epochs = 20       # Número de iterações de treinamento

# Cria e treina o modelo Doc2Vec
model = Doc2Vec(vector_size=vector_size, window=window_size, dm=1, min_count=2, workers=cores, epochs=epochs)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

# Exporta o modelo treinado para um arquivo
model.save('doc2vec_models/meu_modelo_doc2vec.pt')

print("Modelo Doc2Vec treinado e salvo com sucesso!")
