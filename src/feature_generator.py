from src.features.feature_basic import *

#filename =

basic_features = [
    CharacterCount(),
    WordCount()
]

feature_list = []
feature_list.extend(basic_features)

with open(filename, "w") as file:
    for feature in feature_list:
        print(feature.generate())


