from src.features.feature_basic import *
from src.features.feature_tfidf import *
from src.features.feature_statistics import *
from src.aggregator import *
from sklearn.feature_extraction.text import TfidfVectorizer
from src.features.feature_distance import *
from src.file_util import *

import csv

features_basic = [
    CharacterDiff(),
    WordDiff(),
    DigitDiff()
]

features_distance = [
    EditDistance(),
    SentenceJaccard(),
    SentenceDice(),
    CompressionDistance(),
    NgramDistance(),
]


####### TfIdf #########
train = load_train_corpas(1)
tfIdf_model = TfidfVectorizer()
tfIdf_model.fit_transform(train)
svd_model = TruncatedSVD()
features_tfidf = [
    tfIdf(tfIdf_model),
    tfIdfSvd(tfIdf_model, svd_model)
]

##### stat #######
df_dict = create_df_dict(train)
features_stat = [
    CooccuranceCount(0.99, basic_aggregator),
    IdfCount(df_dict=df_dict,document_num=len(train),aggregator=basic_aggregator,threshold=0.99)
]

feature_list = []
#feature_list.extend(features_basic)
#feature_list.extend(features_distance)
#feature_list.extend(features_tfidf)
feature_list.extend(features_stat)

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