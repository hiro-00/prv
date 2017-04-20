import csv
import gensim

from src.file_util import *
data_dir = "../../data/"
model_dir = "../../model/"


def word2vec():
    document = []
    with open(data_dir + "corpas.txt", "r") as f:
        i = 0
        for row in f:
            i += 1
            document.append(row.split())
            if i == 100:
                pass#break


    model = gensim.models.Word2Vec(document, min_count=1)
    #print(model.wv['google'])
    model.save(model_dir + "word2vec_train")

def glove2word2vec():
    glove2 = "../../model/glove.6B.50d.txt"
    w2vec =  "../../model/glove.6B.50d.bin"
    gensim.scripts.glove2word2vec.glove2word2vec(glove2, w2vec)

word2vec()
