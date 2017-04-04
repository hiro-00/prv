from src.features.feature_basic import *
from src.features.feature_tfidf import *
from src.features.feature_statistics import *
from src.features.feature_word2vec import *
from src.aggregator import *
from sklearn.feature_extraction.text import TfidfVectorizer
from src.features.feature_distance import *
from src.file_util import *
import gensim

import csv

def use_basic():
    features_basic = [
        CharacterDiff(),
        WordDiff(),
        DigitDiff()
    ]
    return features_basic

def use_distance():
    features_distance = [
        EditDistance(),
        SentenceJaccard(),
        SentenceDice(),
        CompressionDistance(),
        NgramDistance(),
    ]
    return features_distance

def use_tfidf():
    train = load_train_corpas()
    tfIdf_model = TfidfVectorizer()
    tfIdf_model.fit_transform(train)
    svd_model = TruncatedSVD()
    features_tfidf = [
        tfIdf(tfIdf_model),
        tfIdfSvd(tfIdf_model, svd_model)
    ]
    return features_tfidf

def use_stat():
    train = load_train_corpas()
    df_dict = create_df_dict(train)
    features_stat = [
        CooccuranceCount(0.99, basic_aggregator),
        IdfCount(df_dict=df_dict,document_num=len(train),aggregator=basic_aggregator,threshold=0.99)
    ]
    return features_stat

#### word2vec #####
quora_model = gensim.models.KeyedVectors.load("../model/word2vec")
#quora_ms_model = gensim.models.Word2Vec.load("../model/quora_ms")
#google_model = gensim.models.KeyedVectors.load_word2vec_format("../model/GoogleNews-vectors-negative300.bin", binary=True)
glove_6b_50d = gensim.models.KeyedVectors.load_word2vec_format("../model/glove.6B.50d.bin")
features_word2vec = [
    Word2VecDiff(quora_model),
   # Word2VecDiff(google_model, 300),
    Word2VecDiff(glove_6b_50d, 50),
]

feature_list = []
#feature_list.extend(use_basic())
#feature_list.extend(use_distance())
#feature_list.extend(use_tfidf())
#feature_list.extend(use_stat())
feature_list.extend(features_word2vec)

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