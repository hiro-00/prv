from src.features.feature_basic import *
from src.features.feature_tfidf import *
from src.features.feature_statistics import *
from src.features.feature_word2vec import *
from src.features.feature_distance import *
from src.features.feature_intersect import *
from src.aggregator import *
from src.file_util import *
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
import csv


def use_basic():
    features = [
        CharacterDiff(),
        WordDiff(),
        DigitDiff()
    ]
    return features


def use_distance():
    features = [
        EditDistance(),
        SentenceJaccard(),
        LooseJaccard(0.9),
        LooseJaccard(0.8),
        LooseCount(0.9),
        LooseCount(0.8),
        SentenceDice(),
        CompressionDistance(),
        NgramDistance(),
        EditAggregate(basic_aggregator)
    ]
    return features


def use_tfidf():
    train = load_train_corpas()
    svd = TruncatedSVD()
    features = []
    for ngram in range(1, 4):
        word_tfidf = TfidfVectorizer(min_df=3,
                                    max_df=0.75,
                                    max_features=None,
                                    norm="l2",
                                    strip_accents="unicode",
                                    analyzer="word",
                                    token_pattern=r"\w{1,}",
                                    ngram_range=(1, ngram),
                                    use_idf=1,
                                    smooth_idf=1,
                                    sublinear_tf=1)
        char_tfidf = TfidfVectorizer(min_df=3,
                                    max_df=0.75,
                                    max_features=None,
                                    norm="l2",
                                    strip_accents="unicode",
                                    analyzer="char",
                                    token_pattern=r"\w{1,}",
                                    ngram_range=(1, ngram),
                                    use_idf=1,
                                    smooth_idf=1,
                                    sublinear_tf=1)
        word_tfidf.fit_transform(train)
        char_tfidf.fit_transform(train)
        features.append(Tfidf(word_tfidf))
        features.append(Tfidf(char_tfidf))
        features.append(TfidfSvd(word_tfidf, svd))
        features.append(TfidfSvd(char_tfidf, svd))
    return features


def use_stat():
    train = load_train_corpas()
    df_dict = create_df_dict(train)
    features = [
        CooccuranceCountAgg(0.99, basic_aggregator),
        IdfCountAgg(df_dict=df_dict,document_num=len(train),aggregator=basic_aggregator,threshold=0.99)
    ]
    return features


def use_intersect():
    features = [
        CooccuranceCount(0.8),
        CooccuranceCount(0.98),
        IntersectCount(0.8),
        IntersectCount(0.98)
    ]
    return features


def use_word2vec():
    #### word2vec #####
    quora_model = gensim.models.KeyedVectors.load("../model/word2vec")
    #quora_ms_model = gensim.models.Word2Vec.load("../model/quora_ms")
    #google_model = gensim.models.KeyedVectors.load_word2vec_format("../model/GoogleNews-vectors-negative300.bin", binary=True)
    glove_6b_50d = gensim.models.KeyedVectors.load_word2vec_format("../model/glove.6B.50d.bin")
    features = [
        Word2VecDiff(quora_model),
       # Word2VecDiff(google_model, 300),
        Word2VecDiff(glove_6b_50d, 50),
    ]
    return features


def main():
    feature_list = []
    #feature_list.extend(use_basic())
    #feature_list.extend(use_distance())
    feature_list.extend(use_tfidf())
    #feature_list.extend(use_stat())
    #feature_list.extend(use_word2vec())
    #feature_list.extend(use_intersect())

    filename = "../data/train_processed.csv"
    features = []
    row_num = 0
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter=',', quotechar='"')
        next(reader)
        for row in reader:
            row_num += 1
            if row_num == 100:
                break
            s1 = row[4]
            s2 = row[5]
            row_feature = []
            for feature_generator in feature_list:
                row_feature.extend(feature_generator.gen(s1, s2))
            features.append(row_feature)

    print(features)

if __name__ == "__main__":
    main()