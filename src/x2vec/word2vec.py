from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .. import config
from src.dist_utils import *

class Word2Vec():
    def __init__(self, model, size = None):
        self.model = model
        if size == None:
            size = self.model.vector_size
        self.vector_size = size

    def _get_valid_word_list(self, text):
        return [w for w in text.lower().split(" ") if w in self.model]

    def _get_importance(self, text1, text2):
        len_prev_1 = len(text1.split(" "))
        len_prev_2 = len(text2.split(" "))
        len1 = len(self._get_valid_word_list(text1))
        len2 = len(self._get_valid_word_list(text2))
        imp = default_divide(len1+len2, len_prev_1+len_prev_2)
        return imp

    def _get_n_similarity(self, text1, text2):
        lst1 = self._get_valid_word_list(text1)
        lst2 = self._get_valid_word_list(text2)
        if len(lst1) > 0 and len(lst2) > 0:
            return self.model.n_similarity(lst1, lst2)
        else:
            return config.MISSING_VALUE_NUMERIC

    def _get_n_similarity_imp(self, text1, text2):
        sim = self._get_n_similarity(text1, text2)
        imp = self._get_importance(text1, text2)
        return sim * imp

    def _get_centroid_vector(self, text):
        lst = self._get_valid_word_list(text)
        centroid = np.zeros(self.vector_size)
        for w in lst:
            centroid += self.model[w]
        if len(lst) > 0:
            centroid /= float(len(lst))
        return centroid

    def get_centroid_vdiff(self, text1, text2):
        centroid1 = self._get_centroid_vector(text1)
        centroid2 = self._get_centroid_vector(text2)
        return abs(centroid1 - centroid2)

    def get_centroid_rmse(self, text1, text2):
        centroid1 = self._get_centroid_vector(text1)
        centroid2 = self._get_centroid_vector(text2)
        return rmse(centroid1, centroid2)

    def get_centroid_rmse_imp(self, text1, text2):
        return self.get_centroid_rmse(text1, text2) * self.get_importance(text1, text2)

