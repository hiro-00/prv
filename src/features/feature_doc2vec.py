import gensim
from src.x2vec.doc2vec_util import gen_label
from sklearn.metrics.pairwise import cosine_similarity
from src.dist_utils import *

class Doc2VecDiff():
    def __init__(self, model, label_dict):
        self.model = model
        self.label_dict = label_dict

    def gen(self, s1, s2):
        v1 = self.model.docvecs[self.label_dict[s1]]
        v2 = self.model.docvecs[self.label_dict[s2]]
        return abs(v1 - v2)

class Doc2VecCosine():
    def __init__(self, model, label_dict):
        self.model = model
        self.label_dict = label_dict

    def gen(self, s1, s2):
        v1 = self.model.docvecs[self.label_dict[s1]]
        v2 = self.model.docvecs[self.label_dict[s2]]
        return cosine_similarity(v1, v2)


class Doc2VecRmse():
    def __init__(self, model, label_dict):
        self.model = model
        self.label_dict = label_dict

    def gen(self, s1, s2):
        v1 = self.model.docvecs[self.label_dict[s1]]
        v2 = self.model.docvecs[self.label_dict[s2]]
        return rmse(v1, v2)

if __name__ == "__main__":
    model = gensim.models.Doc2Vec.load("../../model/doc2vec")
    label_dict = gen_label()
    #print(label_dict)#["What can make Physics easy to learn?,How can you make physics easy to learn?"])
    feature = Doc2VecDiff(model, label_dict)
    print(feature.gen("How do I read and find my YouTube comments?",
                      "How can you make physics easy to learn?"))