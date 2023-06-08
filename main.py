import sys
import itertools
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import Lasso
from Preprocessor import preprocessor, preprocess_labels_q1, load_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeClassifier
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessClassifier

REGRESSION = "1"

CLASSIFICATION = "0"

PREDICTION_TYPE = 1
X_TRAIN_FILE = 2
Y_TRAIN_FILE = 3
X_TEST_FILE = 4

NUM_OF_METASTASES = 11
K = 10
ALPHA = 1

COLUMN_NAMES_SPACE = [
    'ADR - Adrenals',
    'BON - Bones',
    'BRA - Brain',
    'HEP - Hepatic',
    'LYM - Lymphnodes',
    'MAR - BoneMarrow',
    'OTH - Other',
    'PER - Peritoneum',
    'PLE - Pleura',
    'PUL - Pulmonary',
    'SKI - Skin'
]


def flatten(ls):
    """
    flatten a nested list
    """
    flat_ls = list(itertools.chain.from_iterable(ls))
    return flat_ls


class Encode_Multi_Hot:
    """
    change the variable length format into a
    fixed size one hot vector per each label
    """

    def __init__(self):
        """
        init data structures
        """
        self.label_to_ind = {}
        self.ind_to_label = {}
        self.num_of_label = None

    def fit(self, raw_labels):
        """
        learn about possible labels
        """
        # get a list of unique values in df
        labs = list(set(flatten(raw_labels)))
        inds = list(range(len(labs)))
        self.label_to_ind = dict(zip(labs, inds))
        self.ind_to_label = dict(zip(inds, labs))
        self.num_of_label = len(labs)

    def enc(self, raw_label):
        """
        encode variable length category list into multiple hot
        """
        multi_hot = np.zeros(self.num_of_label)
        for lab in raw_label:
            cur_ind = self.label_to_ind[lab]
            multi_hot[cur_ind] = 1
        return multi_hot


def parse_df_labels(df):
    """
    Return a dictionary of response name and values from df
    """
    assert (len(df.columns) == 1)
    resp = df.columns[0]
    ls = [eval(val) for val in df[resp]]
    ret_dict = {"resp": resp, "vals": ls}
    return ret_dict


def indicator_matrix_to_lists(dummies, col_names) -> pd.DataFrame:
    true_columns = []
    dummies = pd.DataFrame(dummies, columns=col_names)
    for index, row in dummies.iterrows():
        true_values = row[row == 1].index.tolist()
        true_columns.append(true_values)
    dummies['אבחנה-Location of distal metastases'] = true_columns
    return dummies['אבחנה-Location of distal metastases']


def predicting_metastases_v1(X_train, X_test, y_train, col_names):
    random_forest = ensemble.RandomForestClassifier(max_depth=10, random_state=42, class_weight="balanced")
    classifier = OneVsRestClassifier(estimator=random_forest)
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)

    pred_df = indicator_matrix_to_lists(pred, col_names)
    pred_df.to_csv("submission/predictions/1.csv", index=False)


def classification_preprocess_y(y_file):
    dfy_train = parse_df_labels(pd.read_csv(y_file, keep_default_na=False))
    enc = Encode_Multi_Hot()
    dfy_train_vals = dfy_train["vals"]
    enc.fit(dfy_train_vals)  # get all possible labels
    dfy_train_multi_hot = np.array([enc.enc(val) for val in dfy_train_vals])
    col_names = [enc.ind_to_label[i] for i in range(len(enc.ind_to_label))]
    return dfy_train_multi_hot, col_names


def regression_preprocess_y(y_file: str) -> Tuple[np.array, List[str]]:
    dfy_train = parse_df_labels(pd.read_csv(y_file, keep_default_na=False, dtype={'אבחנה-Tumor size': str}))
    dfy_train_vals = np.array(dfy_train["vals"])
    col_names = []

    return dfy_train_vals, col_names


def load_files_to_array(argv):
    # load files
    dfX_train = preprocessor(load_data(argv[X_TRAIN_FILE]))
    dfX_test = preprocessor(load_data(argv[X_TEST_FILE]))

    # get labels
    if (argv[PREDICTION_TYPE] == "0"):
        dfy_train, col_names = classification_preprocess_y(argv[Y_TRAIN_FILE])
        X_train, y_train = dfX_train.to_numpy(), dfy_train.astype(int)
    else:
        dfy_train, col_names = regression_preprocess_y(argv[Y_TRAIN_FILE])
        X_train, y_train = dfX_train.to_numpy(), dfy_train.astype(float)

    X_test = dfX_test.to_numpy()
    return X_train, y_train, X_test, col_names


def predicting_tumer_size_v1(X_train, X_test, y_train):
    lasso = Lasso(ALPHA)
    lasso.fit(X_train, y_train)
    pred = lasso.predict(X_test)
    pd.DataFrame(pred).to_csv("submission/predictions/2.csv", header=["אבחנה-Tumor size"], index=False)


# def predicting_tumer_size_v1(X_train, X_test, y_train, y_test):
#     # TODO: remove
#
#     gold_fn = "tests_sets/test1_label_1.csv"
#     gold_labels = parse_df_labels(pd.read_csv(gold_fn, keep_default_na=False, dtype={'אבחנה-Tumor size': str}))
#     gold_vals = gold_labels["vals"]
#
#     #min loss: 3.745278371442692, argmin: 2650.0
#     alphas = np.linspace(2650, 3000, 40)
#     loss = np.zeros(len(alphas))
#     for i, a in enumerate(alphas):
#         lasso = Ridge(alpha=a)
#         lasso.fit(X_train, y_train)
#         pred = lasso.predict(X_test)
#         loss[i] = mean_squared_error(y_true=gold_vals, y_pred=pred)
#
#     print(f"min loss: {np.min(loss)}, argmin: {alphas[np.argmin(loss)]}")
def run_predict_q1(X_train_file, y_train_file, X_test_file, param):
    X_train, y_train, X_test, col_names = load_files_to_array(X_train_file, y_train_file, X_test_file)

    # Q1
    return predicting_metastases_v1(X_train, X_test, y_train, col_names, param)


if __name__ == '__main__':
    # TODO: VALIDATE ARGS

    X_train, y_train, X_test, col_names = load_files_to_array(sys.argv)
    #  TODO: remove - its for debugging
    # train_after_parse = indicator_matrix_to_lists(y_test, col_names)
    # pd.DataFrame(train_after_parse).to_csv("tree_pred.csv", index=False)

    # Q1
    if (sys.argv[PREDICTION_TYPE] == CLASSIFICATION):
        predicting_metastases_v1(X_train, X_test, y_train, col_names)

    # Q2
    if (sys.argv[PREDICTION_TYPE] == REGRESSION):
        predicting_tumer_size_v1(X_train, X_test, y_train)


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
    print("number of false negative ", wrong - false_positive)
    print("number of true positive ", )
    print("predicted negative ", len(pred_negative_index[0]))
