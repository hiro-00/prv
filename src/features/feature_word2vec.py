from src.x2vec import word2vec
import gensim


class Word2VecDiff():
    def __init__(self, model, size = None):
        self.model = word2vec.Word2Vec(model, size)

    def gen(self,s1, s2):
        return self.model.get_centroid_vdiff(s1,s2)


class Word2VecRmse():
    def __init__(self, model, size = None):
        self.model = word2vec.Word2Vec(model, size)

    def gen(self,s1, s2):
        return [self.model.get_centroid_rmse(s1,s2), self.model.get_centroid_rmse_imp(s1,s2)]



class Word2VecSim():
    def __init__(self, model, size = None):
        self.model = word2vec.Word2Vec(model, size)

    def gen(self,s1, s2):
        return [self.model.get_n_similarity(s1,s2),self.model.get_n_similarity_imp(s1,s2)]


class Word2VecAveVdiff():
    def __init__(self, model, size = None):
        self.model = word2vec.Word2Vec(model, size)

    def gen(self,s1, s2):
        return [self.model.get_average_diff(s1,s2)]

