import itertools
import os
import sys
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.linear_model import Ridge
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier

from hackathon_code.Preprocessor import load_data
from hackathon_code.Preprocessor import preprocessor

USAGE_MSG = """Usage: <program_name> <0/1> <train features file name> <train labels file name> <predict set>"""
CLASSIFICATION = "0"
REGRESSION = "1"

PREDICTION_TYPE = 1
X_TRAIN_FILE = 2
Y_TRAIN_FILE = 3
X_TEST_FILE = 4

NUM_OF_METASTASES = 11
K = 10
ALPHA = 2786.44



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


def predicting_metastases(X_train, X_test, y_train, col_names):
    estimator = ensemble.AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=60)
    classifier = OneVsRestClassifier(estimator=estimator)
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)

    pred_df = indicator_matrix_to_lists(pred, col_names)
    pred_df.to_csv("./task_2/predictions/1.csv", index=False)


def classification_preprocess_y(y_file):
    dfy_train = parse_df_labels(pd.read_csv(y_file, keep_default_na=False, dtype={'אבחנה-Tumor size': str}))
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
    dfX_test = dfX_test.reindex(columns=dfX_train.columns, fill_value=0)

    # get labels
    if (argv[PREDICTION_TYPE] == "0"):
        dfy_train, col_names = classification_preprocess_y(argv[Y_TRAIN_FILE])
        X_train, y_train = dfX_train.to_numpy(), dfy_train.astype(int)
    else:
        dfy_train, col_names = regression_preprocess_y(argv[Y_TRAIN_FILE])
        X_train, y_train = dfX_train.to_numpy(), dfy_train.astype(float)

    X_test = dfX_test.to_numpy()
    return X_train, y_train, X_test, col_names


def predicting_tumer_size(X_train, X_test, y_train):
    estimator = Ridge(ALPHA)
    estimator.fit(X_train, y_train)

    pred = estimator.predict(X_test)
    pred = np.where(pred < 0, 0, pred)
    pd.DataFrame(pred).to_csv("./task_2/predictions/2.csv", header=["אבחנה-Tumor size"], index=False)


if __name__ == '__main__':
    """
    Usage:
    <program_name> <0/1>  <train features file name> <train labels file name> <predict set>
    """
    if not len(sys.argv) == 5:
        print(USAGE_MSG, file=sys.stderr)
        exit(1)
    if sys.argv[PREDICTION_TYPE] != CLASSIFICATION and sys.argv[PREDICTION_TYPE] != REGRESSION:
        print(f"Usage: this argument need to be in [{CLASSIFICATION},{REGRESSION}] only \n "
              f"0  - to run the first task \n "
              f"1 - to run the second task", file=sys.stderr)
        exit(1)
    for i in range(3, 5):
        if not os.path.exists(sys.argv[i]):
            print(f"Usage: invalid path please enter an existing file. {sys.argv[i]} does not exist\n", file=sys.stderr)
            exit(1)

    X_train, y_train, X_test, col_names = load_files_to_array(sys.argv)

    # Q1
    if (sys.argv[PREDICTION_TYPE] == CLASSIFICATION):
        predicting_metastases(X_train, X_test, y_train, col_names)

    # Q2
    if (sys.argv[PREDICTION_TYPE] == REGRESSION):
        predicting_tumer_size(X_train, X_test, y_train)
