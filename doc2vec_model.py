import gensim
import load
import os
import re
from nltk.tokenize import RegexpTokenizer
# from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim.models.doc2vec import TaggedDocument
from functions import *

# Repositório:
# http://nilc.icmc.usp.br/nilc/index.php/repositorio-de-word-embeddings-do-nilc#
# Código:
# https://ireneli.eu/2016/07/27/nlp-05-from-word2vec-to-doc2vec-a-simple-example-with-gensim/

# Seleciona uma lista de documentos para serem utilizadas com o doc2vec
def get_doc_list(folder_name):
    doc_list = []
    file_names = []
    file_list = [folder_name+'/'+name for name in os.listdir(folder_name) if name.endswith('txt')]
    for file in file_list:
        st = open(file, 'r', encoding='utf-8').read()
        file_names.append(file)
        doc_list.append(st)
    print ('Found %s documents under the dir %s .....'%(len(file_list),folder_name))
    return doc_list, file_names
 
def get_doc(folder_name):
 
    print("\nEntrando na função get_doc")
    doc_list, file_names_list = get_doc_list(folder_name)
    tokenizer = RegexpTokenizer(r'\w+')
    # en_stop = get_stop_words('en')
    p_stemmer = PorterStemmer()
    taggeddoc = []
    texts = []

    for index, i in enumerate(doc_list):
        print("Passando pelo documento ", file_names_list[index])

        # for tagged doc
        wordslist = []
        tagslist = []
 
        # clean and tokenize document string
        try:
            raw = i.lower()
            tokens = tokenizer.tokenize(raw)
    
            # remove stop words from tokens
            # stopped_tokens = [i for i in tokens if not i in en_stop]
    
            # remove numbers
            number_tokens = [re.sub(r'[\d]', ' ', i) for i in tokens]
            number_tokens = ' '.join(number_tokens).split()
    
            # stem tokens
            stemmed_tokens = [p_stemmer.stem(i) for i in number_tokens]
            # remove empty
            length_tokens = [i for i in stemmed_tokens if len(i) > 1]
            # add tokens to list
            texts.append(length_tokens)
    
            td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(stemmed_tokens))).split(),str(index))
            # for later versions, you may want to use: td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(stemmed_tokens))).split(),[str(index)])
            taggeddoc.append(td)

        except:
            print("Encontrou um elemento muito grande: ", len(i))
            print("Nome do arquivo: ", file_names_list[index])
 
    return taggeddoc

# ===============================================================================================================

print("Carregando e formatando os dados...")
documents = get_doc('documentos_doc2vec')
print ('\nDados carregados!\n')
 
print (len(documents),type(documents))
 
# build the model
model = gensim.models.Doc2Vec(documents, dm = 0, alpha=0.025, vector_size= 20, min_alpha=0.025, min_count=0)
 
# start training
# for epoch in range(200):
#     if epoch % 20 == 0:
#         print ('Now training epoch %s'%epoch)
model.build_vocab(documents)
model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
model.alpha -= 0.002  # decrease the learning rate
model.min_alpha = model.alpha  # fix the learning rate, no decay
 
# shows the similar words
# print("\nMostrando as palavras similares")
# print (model.wv.most_similar('filme'))
 
# print("\nMostrando o que foi aprendido sobre a palavra filme: ")
# shows the learnt embedding
# print (model['filme'])
 
# shows the similar docs with id = 2
# print (model.docvecs.most_similar(str(2)))

print("\nSalvando o modelo")
model.save('doc2vec_models/trained.model')
model.save_word2vec_format('doc2vec_models/trained.word2vec')

# ===============================================================================================================

# load the word2vec
# word2vec = gensim.models.Doc2Vec.load_word2vec_format('save/trained.word2vec')
# print (word2vec['música'])
 
# load the doc2vec
model = gensim.models.Doc2Vec.load('doc2vec_models/trained.model')
docvecs = model.docvecs
# print (docvecs[str(3)])

print("\nFIM")

# def plotWords():
#     #get model, we use w2v only
#     w2v,d2v=useModel()
 
#     words_np = []
#     #a list of labels (words)
#     words_label = []
#     for word in w2v.vocab.keys():
#         words_np.append(w2v[word])
#         words_label.append(word)
#     print('Added %s words. Shape %s'%(len(words_np),np.shape(words_np)))
 
#     pca = decomposition.PCA(n_components=2)
#     pca.fit(words_np)
#     reduced= pca.transform(words_np)
 
#     # plt.plot(pca.explained_variance_ratio_)
#     for index,vec in enumerate(reduced):
#         # print ('%s %s'%(words_label[index],vec))
#         if index <100:
#             x,y=vec[0],vec[1]
#             plt.scatter(x,y)
#             plt.annotate(words_label[index],xy=(x,y))
#     plt.show()