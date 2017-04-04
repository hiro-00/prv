from src.x2vec import word2vec
import gensim


class Word2VecDiff():
    def __init__(self, model, size = None):
        self.model = word2vec.Word2Vec(model, size)

    def gen(self,s1, s2):
        return self.model.get_centroid_vdiff(s1,s2)

