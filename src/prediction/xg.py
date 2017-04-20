
import pickle
import random
import xgboost as xgb
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

from sklearn.cross_validation import train_test_split
import pandas as pd


def resample(train, label):
    df_train = pd.DataFrame(data=train)
    df_label = pd.DataFrame(label)
    pos_train = df_train[df_label.values == 1]
    neg_train = df_train[df_label.values == 0]
    print(len(pos_train))
    print(len(neg_train))

    p = 0.160
    scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
    while scale > 1:
        neg_train = pd.concat([neg_train, neg_train])
        scale -= 1
    neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])

    x_train = pd.concat([pos_train, neg_train]).as_matrix()
    label = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()

    print(x_train.shape)
    print(len(label))
    return x_train, label

def create_xgb_model(x, y, seed_val=0):
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=4242)

    param = {}
    param['objective'] = 'binary:logistic'
    param['eta'] = 0.02
    param['max_depth'] = 4
    #param['silent'] = 1
    param['eval_metric'] = "logloss"
    param['seed'] = seed_val
    num_rounds = 2000
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dvalid = xgb.DMatrix(x_valid, label=y_valid)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    model = xgb.train(params=param, dtrain=dtrain, evals=watchlist,num_boost_round=num_rounds,verbose_eval=10)
    return model


def cv(train, label):
    av_score = 0
    fold = 20
    kf = KFold( n_splits=fold, shuffle=True)
    for train_index, test_index in kf.split(train, label):
        train_X = train#[train_index]
        train_y = label#[train_index]
        test_X = train#[test_index]
        test_y = label#[test_index]

        print('run model')
        model = create_xgb_model(train_X, train_y)
        predicted = model.predict(xgb.DMatrix(train_X), 4242)
        score = log_loss(train_y, predicted)
        print(score)
        predicted = model.predict(xgb.DMatrix(test_X))
        print(predicted)
        score = log_loss(test_y, predicted)
        av_score += score
        print(score)
        print("--")
    print(av_score/fold)

feature_file_path = "../../features/"
feature_files =[
    "train_basic.csv",
     "train_distance.csv",
     "train_intersect.csv",
     "train_stat.csv",
     "train_tfidf.csv",
     "train_tfidf_rmse.csv",
     "train_tfidf_shared.csv"
]

train = None
for file in feature_files:
    path = feature_file_path + file
    data = np.loadtxt(path, delimiter=",")
    print(data.shape)
    if train is None:
        train = data
    else:
        train = np.hstack((train, data))
    print(train.shape)


label = np.loadtxt("../../data/label.txt")

train, label = resample(train, label)

cv(train, label)


