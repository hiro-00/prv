from src.dist_utils import loose_match
from src.nlp_utils import *
from src import config
from collections import defaultdict
import numpy as np

def create_df_dict(corpas):
    df_dict = defaultdict(int)
    for sentence in corpas:
        word_set = set()
        for word in sentence.split():
            word_set.add(word)
        for word in word_set:
            df_dict[word] += 1
    return df_dict

def get_idf(N, df_dict, word):
    return np.log((N - df_dict[word] + 0.5)/(df_dict[word] + 0.5))

class CooccuranceCountAgg():
    def __init__(self, threshold, aggregator):
        self.threshold = threshold
        self.aggregator = aggregator

    def gen(self, s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        var_list = []
        for n in range(1,4):
            ngram1 = ngram(s1, n + 1)
            ngram2 = ngram(s2, n + 1)
            for w1 in ngram1:
                count = 0
                for w2 in ngram2:
                    if loose_match(w1, w2, self.threshold):
                        count += 1
                var_list.append(count)
        if len(var_list) == 0:
            var_list = [config.MISSING_VALUE_NUMERIC]
        return self.aggregator(var_list)

class IdfCountAgg():
    def __init__(self, document_num, threshold, aggregator, df_dict):
        self.threshold = threshold
        self.aggregator = aggregator
        self.df_dict = df_dict
        self.document_num = document_num

    def gen(self, s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        var_list = []
        for n in range(1):
            ngram1 = ngram(s1, n + 1)
            ngram2 = ngram(s2, n + 1)
            for w1 in ngram1:
                count = 0
                for w2 in ngram2:
                    if loose_match(w1, w2, self.threshold):
                        count += 1
                var_list.append(count * get_idf(self.document_num, self.df_dict, w1))
        if len(var_list) == 0:
            var_list = [config.MISSING_VALUE_NUMERIC]
        return self.aggregator(var_list)