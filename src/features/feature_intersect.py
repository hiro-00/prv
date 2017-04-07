from src.dist_utils import loose_match
from src.nlp_utils import *
from ..dist_utils import *


class CooccuranceCount():
    def __init__(self, threshold):
        self.threshold = threshold

    def gen(self, s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        var_list = []
        for n in range(1,4):
            ngram1 = ngram(s1, n + 1)
            ngram2 = ngram(s2, n + 1)
            count = .0
            for w1 in ngram1:
                for w2 in ngram2:
                    if loose_match(w1, w2, self.threshold):
                        count += 1
                        break
            var_list.append(count)
        var_ratio = [default_divide(var, len(s1)) for var in var_list]
        var_list.extend(var_ratio)
        return var_list


class IntersectCount():
    def __init__(self, threshold):
        self.threshold = threshold

    def gen(self, s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        var_list = []
        for n in range(1, 4):
            ngram1 = ngram(s1, n + 1)
            ngram2 = ngram(s2, n + 1)
            count = 0
            for w1 in ngram1:
                for w2 in ngram2:
                    if loose_match(w1, w2, self.threshold):
                        count += 1
            var_list.append(count)
        var_ratio = [default_divide(var, len(s1)) for var in var_list]
        var_list.extend(var_ratio)
        return var_list