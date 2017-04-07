from nltk.corpus import wordnet as wn


class WordnetBasic():
    def __init__(self):
        self.metrics = [lambda syn1, syn2: wn.wup_similarity(syn1, syn2),
                        lambda syn1, syn2: wn.lch_similarity(syn1, syn2),
                        lambda syn1, syn2: wn.path_similarity(syn1, syn2)]

    def _maximum_similarity_for_two_synset_list(self, metric, syn_list1, syn_list2):
        s = 0.
        if syn_list1 and syn_list2:
            for syn1 in syn_list1:
                for syn2 in syn_list2:
                    try:
                        _s = metric(syn1, syn2)
                    except:
                        _s = -1
                    if _s and _s > s:
                        s = _s
        return s

    def gen(self, s1, s2):
        words1 = s1.split()
        words2 = s2.split()
        if len(words1) < len(words2):
            words2, words1 = words1, words2
        words1 = [wn.synsets(word) for word in words1]
        words2 = [wn.synsets(word) for word in words2]
        features = []
        for metric in self.metrics:
            for offset in range(1, 3):
                ave_sim = 0
                for center in range(0, len(words1)):
                    word1 = words1[center]
                    cands = [words2[min(max(0,pos),len(words2)-1)] for pos in range(center - offset, center + offset + 1)]
                    max_sim = 0
                    for cand in cands:
                        max_sim = max(max_sim, self._maximum_similarity_for_two_synset_list(metric, word1, cand))
                    ave_sim += max_sim
                features.append(ave_sim/len(words1))
        return features

if __name__ == "__main__":
    model = WordnetBasic()
    print(model.gen("hello world", "hi"))