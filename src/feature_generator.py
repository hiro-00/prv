from src.features.feature_basic import *
from src.features.feature_distance import *
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

feature_list = []
feature_list.extend(features_distance)

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
        s1 = row[2]
        s2 = row[3]
        row_feature = []
        for feature_generator in feature_list:
            row_feature.extend(feature_generator.gen(s1, s2))
        features.append(row_feature)

print(features)