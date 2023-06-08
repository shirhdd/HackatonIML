import sys
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import Lasso
from Preprocessor import preprocessor, preprocess_labels_q1, load_data
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.tree import DecisionTreeClassifier


NUM_OF_METASTASES = 11
K = 10
ALPHA = 0.3


def loss(y_true, y_pred):
    res = f1_score(y_true, y_pred, average='micro')
    res += f1_score(y_true, y_pred, average='macro')
    return res


# def predicting_metastases_v1(X_train, X_test, y_train, y_test):
#     tree = DecisionTreeClassifier()
#     tree.fit(X_train, y_train)
#     pred = tree.predict(X_test)
#     result_eval(pred, y_test)
#     pd.DataFrame(pred, columns=["אבחנה-Location of distal metastases"]).to_csv("tree_pred.csv", index=False)


def predicting_metastases_v1(X_train, X_test, y_train, y_test):
    tree = DecisionTreeClassifier()
    multiclass = MultiOutputClassifier(tree, NUM_OF_METASTASES)
    multiclass.fit(X_train, y_train)
    pred = multiclass.predict(X_test)
    # result_eval(pred, y_test)
    pd.DataFrame(pred, columns=["אבחנה-Location of distal metastases"]).to_csv("tree_pred.csv", index=False)


def result_eval(y_pred, y_true):
    y_pred = y_pred.reshape(y_pred.size)
    y_true = y_true.reshape(y_true.size)
    wrong = np.sum(y_pred != y_true)
    pred_positive_index = np.where(y_pred != '[]')
    false_positive = np.sum(y_pred[pred_positive_index] != y_true[pred_positive_index])
    pred_negative_index = np.where(y_pred == '[]')
    print("number of wrong classifications", wrong)
    print("number of correct classification ", y_pred.size - wrong)
    print("number of false positive", false_positive)
    print("number of false negative ", wrong-false_positive)
    print("number of true positive ", )
    print("predicted negative ", len(pred_negative_index[0]))

# def binary_relevance(X_train, X_test, y_train, y_test):
#     # initialize binary relevance multi-label classifier
#     # with a gaussian naive bayes base classifier
#     classifier = BinaryRelevance(GaussianNB())
#     # train
#     classifier.fit(X_train, y_train)
#     # predict
#     y_pred = classifier.predict(X_test)
#     # accuracy
#     print("Micro = ", f1_score(y_test, y_pred, average='micro'))
#     print("Macro = ", f1_score(y_test, y_pred, average='macro'))


def predicting_tumer_size_v1(X_train, X_test, y_train, y_test):
    lasso = Lasso(ALPHA)
    lasso.fit(X_train, y_train)
    print("Score on q2 very basic: ", lasso.score(X_test, y_test))


if __name__ == '__main__':
    dfX_train = preprocessor(load_data(sys.argv[1]))
    dfy_train = pd.read_csv(sys.argv[2])
    dfX_test = preprocessor(load_data(sys.argv[3]))
    dfy_test = pd.read_csv(sys.argv[4])
    X_train, y_train = dfX_train.to_numpy(), dfy_train.to_numpy()
    X_test, y_test = dfX_test.to_numpy(), dfy_test.to_numpy()


    pd.DataFrame(y_test, columns=["אבחנה-Location of distal metastases"]).to_csv("tree_test.csv", index=False)

    # Q1
    predicting_metastases_v1(X_train, X_test, y_train, y_test)

    # Q2
    # predicting_tumer_size_v1(X_train, X_test, y_train, y_test)





