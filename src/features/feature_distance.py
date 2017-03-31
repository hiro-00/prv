import sys
from ..dist_utils import *
from ..nlp_utils import *

class EditDistance():
    def gen(self, s1, s2):
        return edit_dist(s1, s2)

class SentenceJaccard():
    def gen(self, s1, s2):
        return jaccard(list(s1), list(s2))

class SentenceDice():
    def gen(self, s1, s2):
        return dice(list(s1), list(s2))


class CompressionDistance():
    def gen(self, s1, s2):
        return comp_dist(s1, s2)

class NgramDistance():
    def gen(self,s1, s2):
        return [jaccard(ngram(s1, 1), ngram(s2,1)),
                jaccard(ngram(s1, 2), ngram(s2, 2)),
                jaccard(ngram(s1, 3), ngram(s2, 3)),
                dice(ngram(s1, 1), ngram(s2, 1)),
                dice(ngram(s1, 1), ngram(s2, 1)),
                dice(ngram(s1, 1), ngram(s2, 1))]

def main(*argv):
    pass


if __name__ == "__main__":
    main(sys.argv[1])