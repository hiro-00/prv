from src.features.feature_basic import *
from src.features.feature_tfidf import *
from src.features.feature_statistics import *
from src.features.feature_word2vec import *
from src.features.feature_distance import *
from src.features.feature_intersect import *
from src.aggregator import *
from src.x2vec.word2vec import Word2Vec
from src.file_util import *
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
import csv
import pickle
import pandas as pd


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

def use_tfidf_rmse():
    train = load_train_corpas()
    features = []
    for ngram in range(1, 3):
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
        word_tfidf.fit_transform(train)
        features.append(TfidfRmse(word_tfidf))
    return features

def use_tfidf_shared():
    train = load_train_corpas()
    features = []
    for ngram in range(1, 3):
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
        word_tfidf.fit_transform(train)
        features.append(TfidfShared(word_tfidf))
    return features

def use_tfidf_cosine():
    train = load_train_corpas()
    svd = TruncatedSVD()
    features = []
    for ngram in range(1, 3):
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
        word_tfidf.fit_transform(train)
        features.append(TfidfCosine(word_tfidf))

        # char_tfidf = TfidfVectorizer(min_df=3,
        #                             max_df=0.75,
        #                             max_features=None,
        #                             norm="l2",
        #                             strip_accents="unicode",
        #                             analyzer="char",
        #                             token_pattern=r"\w{1,}",
        #                             ngram_range=(1, ngram),
        #                             use_idf=1,
        #                             smooth_idf=1,
        #                             sublinear_tf=1)
        #features.append(Tfidf(char_tfidf))
        #char_tfidf.fit_transform(train)
        #features.append(TfidfSvd(word_tfidf, svd))
        #features.append(TfidfSvd(char_tfidf, svd))
    return features


def use_stat():
    train = load_train_corpas()
    df_dict = create_df_dict(train)
    features = [
        CooccuranceCountAgg(0.99, basic_aggregator),
        CooccuranceCountAgg(0.8, basic_aggregator),
        IdfCountAgg(df_dict=df_dict,document_num=len(train),aggregator=basic_aggregator,threshold=0.99),
        IdfCountAgg(df_dict=df_dict,document_num=len(train),aggregator=basic_aggregator,threshold=0.8)
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


def use_train_word2vec_basic():
    model = gensim.models.KeyedVectors.load("../model/word2vec_train")
    #quora_ms_model = gensim.models.Word2Vec.load("../model/quora_ms")
    #google_model = gensim.models.KeyedVectors.load_word2vec_format("../model/GoogleNews-vectors-negative300.bin", binary=True)
    glove_6b_50d = gensim.models.KeyedVectors.load_word2vec_format("../model/glove.6B.50d.bin")
    features = [
        # Word2VecDiff(model),
        Word2VecRmse(model),
        Word2VecSim(model),
        Word2VecAveVdiff(model)
    ]
    return features

def use_google_word2vec_basic():
    model = gensim.models.KeyedVectors.load_word2vec_format("../model/GoogleNews-vectors-negative300.bin", binary=True)
    size = len(model['google'])
    features = [
        Word2VecRmse(model, size),
        Word2VecSim(model, size),
        Word2VecAveVdiff(model, size)
    ]
    return features


def use_train_word2vec_vdiff():
    #### word2vec #####
    model = gensim.models.KeyedVectors.load("../model/word2vec_train")
    #quora_ms_model = gensim.models.Word2Vec.load("../model/quora_ms")
    #google_model = gensim.models.KeyedVectors.load_word2vec_format("../model/GoogleNews-vectors-negative300.bin", binary=True)
    glove_6b_50d = gensim.models.KeyedVectors.load_word2vec_format("../model/glove.6B.50d.bin")
    features = [
        Word2VecDiff(model),
    ]
    return features


def main():
    feature_list = []
    '''
    feature_list.append(["train_basic.csv", use_basic()])
    feature_list.append(["train_distance.csv", use_distance()])
    feature_list.append(["train_intersect.csv", use_intersect()])
    feature_list.append(["train_stat.csv", use_stat()])
    feature_list.append(["train_tfidf_shared.csv",use_tfidf_shared()])
    feature_list.append(["train_tfidf_rmse.csv",use_tfidf_rmse()])
    feature_list.append(["train_tfidf.csv",use_tfidf_cosine()])
    '''
    feature_list.append(["train_word2vec_google_basic.csv", use_google_word2vec_basic()])
    filename = "../data/train_processed.csv"

    row_num = 0
    train = []
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter=',', quotechar='"')
        for row in reader:
            train.append((row[0], row[1]))

    #train, label = resample(train)

    for feature in feature_list:
        output_filename = "../features/" + feature[0]
        print(output_filename)
        feature_generator_list = feature[1]
        features = []
        for t in train:
                row_feature = []
                for feature_generator in feature_generator_list:
                    row_feature.extend(feature_generator.gen(t[0], t[1]))
                features.append(row_feature)

        with open(output_filename, "w") as file:
            writer = csv.writer(file, lineterminator='\n')
            writer.writerows(features)
    #
    # with open("../features/resampled_label.txt", "w") as f:
    #     for line in label:
    #         print(int(line), file=f)

if __name__ == "__main__":
    main()