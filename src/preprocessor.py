import pandas as pd

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re



df_train = pd.read_csv('../data/train.csv')
df_train = df_train.fillna('joeajeklsdjfasd')

df_train.to_csv('../data/train_processed.csv')