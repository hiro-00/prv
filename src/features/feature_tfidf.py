from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD


class tfIdf():
    def __init__(self, tfidfVectorizer):
        self.tfidf = tfidfVectorizer

    def gen(self,s1, s2):
        v1, v2 = self.tfidf.transform([s1, s2]).toarray()[0:2]
        return cosine_similarity(v1,v2).tolist()[0]

class tfIdfSvd():
    def __init__(self, tfidf, svd):
        self.tfidf = tfidf
        self.svd = svd

    def gen(self,s1, s2):
        v1, v2 = self.tfidf.transform([s1, s2])
        svd_v1 = self.svd.fit_transform(v1)
        svd_v2 = self.svd.fit_transform(v2)
        return cosine_similarity(svd_v1, svd_v2).tolist()[0]

if __name__ == "__main__":
    print(cosine_similarity([1,3,2],[2,0,3])[0])
    # v = TfidfVectorizer()
    # docs=[["hello", "world","hey","lll","a","afda"],["hello", "there"],["a"]]
    # docs = [' '.join(d) for d in docs]
    # features = v.fit_transform(docs)
    # print(v.transform(["hello hey"]))
    # #print(features)
    #
    # #print(v.transform(["hello world"]).toarray())
    # svd = TruncatedSVD(n_components=2)
    # print(features.toarray())
    # print(svd.fit_transform(features))