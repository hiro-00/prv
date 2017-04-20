from sklearn.feature_extraction.text import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from src.dist_utils import rmse, default_divide
import numpy as np


class TfidfCosine():
    def __init__(self, tfidfVectorizer):
        self.tfidf = tfidfVectorizer

    def gen(self,s1, s2):
        v1, v2 = self.tfidf.transform([s1, s2]).toarray()[0:2]
        return cosine_similarity(v1,v2).tolist()[0]

class TfidfRmse():
    def __init__(self, tfidfVectorizer):
        self.tfidf = tfidfVectorizer

    def gen(self,s1, s2):
        v1, v2 = self.tfidf.transform([s1, s2]).toarray()[0:2]
        return [rmse(v1,v2)]

class TfidfShared():
    def __init__(self, tfidfVectorizer):
        self.tfidf = tfidfVectorizer

    def gen(self,s1, s2):
        v1, v2 = self.tfidf.transform([s1, s2]).toarray()[0:2]
        shared_index = np.logical_and(v1!=0, v2!= 0)
        shared_weights = np.sum(v1[shared_index] + v2[shared_index])
        total_weights = np.sum(v1[v1!=0]) + np.sum(v2[v2!=0])
        return [default_divide(shared_weights, total_weights, 1.0)]

class TfidfSvd():
    def __init__(self, tfidf, svd):
        self.tfidf = tfidf
        self.svd = svd

    def gen(self,s1, s2):
        v1, v2 = self.tfidf.transform([s1, s2])
        svd_v1 = self.svd.fit_transform(v1)
        svd_v2 = self.svd.fit_transform(v2)
        return cosine_similarity(svd_v1, svd_v2).tolist()[0]


class TfidfSvdTsne():
    def __init__(self, tfidf, svd):
        self.tfidf = tfidf
        self.svd = svd

    def gen(self, s1, s2):
        v1, v2 = self.tfidf.transform([s1, s2])
        svd_v1 = self.svd.fit_transform(v1)
        svd_v2 = self.svd.fit_transform(v2)
        scaled_v1= StandardScaler().fit_transform(svd_v1)
        scaled_v2 = StandardScaler().fit_transform(svd_v2)
        tsne_v1 = TSNE().fit_transform(scaled_v1)
        tsne_v2 = TSNE().fit_transform(scaled_v2)
        return cosine_similarity(tsne_v1, tsne_v2).tolist()[0]

if __name__ == "__main__":
    #print(cosine_similarity([1,3,2],[2,0,3])[0])
    tfidf = TfidfVectorizer()
    docs=[["hello", "world","hey","lll","a","afda"],["hello", "there"],["a"]]
    docs = [' '.join(d) for d in docs]
    print(docs)
    features = tfidf.fit_transform(docs)
    print(features.toarray())
    s1 = tfidf.transform(["hello hey"]).toarray()[0]
    s2 = tfidf.transform(["a hey"]).toarray()[0]
    print(s1)
    shared_index = np.logical_and(s1!=0, s2!= 0)
    shared_weights = np.sum(s1[shared_index] + s2[shared_index])
    total_weights = np.sum(s1[s1!=0]) + np.sum(s2[s2!=0])
    print(shared_weights)
    print(total_weights)
    #print(shared_index)
    #print(tfidf.inverse_transform(["hello"]))
    #print(features.toarray())
    #print(v.transform(["hello world"]).toarray())
    # svd = TruncatedSVD(n_components=2)
    # model = TfidfSvdTsne(tfidf, svd)
    # model.gen("hello world", "world hey")