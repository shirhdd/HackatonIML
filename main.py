import sys
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import Lasso

NUM_OF_METASTASES = 11
K = 10
ALPHA = 0.3

if __name__ == '__main__':
    # Q1
    dfX, dfy = preprosses(sys.argv[1])
    X, y = pd.to_numpy(dfX), pd.to_numpy(dfy)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    knn = KNeighborsClassifier(K, metric=lambda y_true, y_false:  # loss function of inner learner is sum of both loss in the question
                               f1_score(y_true, y_false, average='micro') + f1_score(y_true, y_false, average='macro'))
    multi_classifier = MultiOutputClassifier(knn, NUM_OF_METASTASES)
    multi_classifier.fit(X_train, y_train)
    multi_classifier.score(X_test, y_test)

    #Q2
    lasso = Lasso(ALPHA)
    lasso.fit(X_train, y_train)





