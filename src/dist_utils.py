import Levenshtein
import lzma
from difflib import SequenceMatcher
import numpy as np

def default_divide(x, y, value = 0.0):
    if y != 0:
        value = x / float(y)
    return value

def rmse(vec1, vec2):
    vdiff = vec1 - vec2
    rmse = np.sqrt(np.mean(vdiff**2))
    return rmse


def loose_match(str1, str2, threshold=1.0):
    assert threshold >= 0.0 and threshold <= 1.0, "Wrong threshold."
    if float(threshold) == 1.0:
        return str1 == str2
    else:
        return (1. - edit_dist(str1, str2)) >= threshold

def loose_jaccard(w1, w2, threshold=1.0):
    match_count = 0
    total_count = 0
    for a in w1:
        for b in w2:
            if loose_match(a,b, threshold):
                match_count += 1
            total_count += 1
    return default_divide(match_count/total_count, 0)

def loose_match_count(w1, w2, threshold=1.0):
    match_count = 0
    for a in w1:
        for b in w2:
            if loose_match(a,b, threshold):
                match_count += 1
    return match_count

def edit_dist(str1, str2):
    try:
        d = Levenshtein.distance(str1, str2)/float(max(len(str1),len(str2)))
    except:
        # https://docs.python.org/2/library/difflib.html
        d = 1. - SequenceMatcher(lambda x: x==" ", str1, str2).ratio()
    return d

def jaccard(w1, w2):
    intersect = set(w1).intersection(set(w2))
    union = set(w1).union(set(w2))
    return default_divide(len(intersect), len(union))

def dice(w1, w2):
    intersect = set(w1).intersection(set(w2))
    return default_divide(2*len(intersect), len(set(w1))*len(set(w2)))

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

