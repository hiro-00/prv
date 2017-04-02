from src.x2vec import word2vec_metrics
import gensim


class Word2VecDiff():
    def __init__(self, model):
        self.model = word2vec_metrics.Word2VecMetrics(model)

    def gen(self,s1, s2):
        return self.model.get_centroid_vdiff(s1,s2)

