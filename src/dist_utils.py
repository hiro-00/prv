import Levenshtein
from difflib import SequenceMatcher

def default_divide(x, y, value = 0.0):
    if y != 0:
        value = x / float(y)
    return value



def edit_dist(str1, str2):
    try:
        d = Levenshtein.distance(str1, str2)/float(max(len(str1),len(str2)))
    except:
        # https://docs.python.org/2/library/difflib.html
        d = 1. - SequenceMatcher(lambda x: x==" ", str1, str2).ratio()
    return d

def jaccard(s1, s2):
    w1 = s1.split()
    w2 = s2.split()
    intersect = set(w1).intersection(w2)
    union = set(w1).union(w2)
    return default_divide(len(intersect), len(union))

def dice(s1, s2):
    w1 = s1.split()
    w2 = s2.split()
    intersect = set(w1).intersection(w2)
    union = set(w1).union(w2)
    return default_divide(intersect, len(union))