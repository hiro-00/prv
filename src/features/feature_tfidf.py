from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD


class tfIdf():
    def __init__(self, tfidf):
        self.tfidf = tfidf

    def gen(self,s1, s2):
        v1 = self.tfidf.transform(s1)
        v2 = self.tfidf.transform(s2)
        return [cosine_similarity(v1,v2)]

class tfIdfSvd():
    def __init__(self, tfidf, svd):
        self.tfidf = tfidf
        self.svd = svd

    def gen(self,s1, s2):
        v1 = self.tfidf.transform(s1)
        v2 = self.tfidf.transform(s2)
        svd_v1 = svd.fit_transform(v1)
        svd_v2 = svd.fit_transform(v2)
        return [cosine_similarity(svd_v1, svd_v2)]

v = TfidfVectorizer()
docs=[["hello", "world","hey","lll","a","afda"],["hello", "there"],["a"]]
docs = [' '.join(d) for d in docs]
features = v.fit_transform(docs)
#print(features)

#print(v.transform(["hello world"]).toarray())
svd = TruncatedSVD(n_components=2)
print(features.toarray())
print(svd.fit_transform(features))