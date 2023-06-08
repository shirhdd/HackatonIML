import sys
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import Lasso
from Preprocessor import load_data, preprocess_labels_q1

NUM_OF_METASTASES = 10
K = 10
ALPHA = 0.3


def predicting_metastases_v1(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(K, metric=lambda y_true, y_false:
                               f1_score(y_true, y_false, average='micro') + f1_score(y_true, y_false, average='macro'))
    multi_classifier = MultiOutputClassifier(knn, NUM_OF_METASTASES)
    multi_classifier.fit(X_train, y_train)
    print("Score on q1 very basic: ", multi_classifier.score(X_test, y_test))


def predicting_tumer_size_v1(X_train, X_test, y_train, y_test):
    lasso = Lasso(ALPHA)
    lasso.fit(X_train, y_train)
    print("Score on q2 very basic: ", lasso.score(X_test, y_test))


if __name__ == '__main__':

    dfX = load_data(sys.argv[1])
    dfy = preprocess_labels_q1(sys.argv[2])
    X, y = np.array(dfX), np.array(dfy)
    dfy.to_csv("preprocess_label_0")
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    #
    # # Q1
    # predicting_metastases_v1(X_train, X_test, y_train, y_test)
    #
    # # Q2
    # predicting_tumer_size_v1(X_train, X_test, y_train, y_test)





