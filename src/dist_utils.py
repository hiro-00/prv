import Levenshtein
import lzma
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

def jaccard(w1, w2):
    intersect = set(w1).intersection(w2)
    union = set(w1).union(w2)
    return default_divide(len(intersect), len(union))

def dice(w1, w2):
    intersect = set(w1).intersection(w2)
    union = set(w1).union(w2)
    return default_divide(intersect, len(union))

def comp_dist(x, y):
    if x == y:
        return 0
    x_b = x.encode('utf-8')
    y_b = y.encode('utf-8')
    l_x = len(lzma.compress(x_b))
    l_y = len(lzma.compress(y_b))
    l_xy = len(lzma.compress(x_b+y_b))
    l_yx = len(lzma.compress(y_b+x_b))
    a = min(l_xy,l_yx)-min(l_x,l_y)
    b = max(l_x,l_y)
    return default_divide(a, b) if a > b else default_divide(b, a)
