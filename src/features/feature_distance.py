import sys
from ..dist_utils import *
from ..nlp_utils import *

class EditDistance():
    def gen(self, s1, s2):
        return [edit_dist(s1, s2)]

class SentenceJaccard():
    def gen(self, s1, s2):
        return [jaccard(s1.split(), s2.split())]

class SentenceDice():
    def gen(self, s1, s2):
        return [dice(s1.split(), s2.split())]


class CompressionDistance():
    def gen(self, s1, s2):
        return [comp_dist(s1, s2)]

class NgramDistance():
    def gen(self,s1, s2):
        return [jaccard(ngram(s1, 1), ngram(s2,1)),
                jaccard(ngram(s1, 2), ngram(s2, 2)),
                jaccard(ngram(s1, 3), ngram(s2, 3)),
                dice(ngram(s1, 1), ngram(s2, 1)),
                dice(ngram(s1, 2), ngram(s2, 2)),
                dice(ngram(s1, 3), ngram(s2, 3))]

class LooseJaccard():
    def __init__(self, threshold=1.0):
        self.threshold = threshold

    def gen(self,s1, s2):
        return [loose_jaccard(ngram(s1, 1), ngram(s2, 1), self.threshold),
                loose_jaccard(ngram(s1, 3), ngram(s2, 3), self.threshold),
                loose_jaccard(ngram(s1, 3), ngram(s2, 3), self.threshold)]


class LooseCount():
    def __init__(self, threshold=1.0):
        self.threshold = threshold

    def gen(self,s1, s2):
        return [loose_match_count(ngram(s1, 1), ngram(s2, 1), self.threshold),
                loose_match_count(ngram(s1, 2), ngram(s2, 2), self.threshold),
                loose_match_count(ngram(s1, 3), ngram(s2, 3), self.threshold)]


class EditAggregate():
    def __init__(self, aggregator):
        self.aggregator = aggregator

    def gen(self, s1, s2):
        candidates = []
        for n in range(2,4):
            for ngram1 in ngram(s1, n):
                for ngram2 in ngram(s2, n):
                    candidates.append(edit_dist(ngram1, ngram2))
        return self.aggregator(candidates)

def main(*argv):
    pass


if __name__ == "__main__":
    main(sys.argv[1])