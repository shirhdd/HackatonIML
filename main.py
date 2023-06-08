import sys
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import Lasso
from Preprocessor import preprocessor, preprocess_labels_q1, load_data
from sklearn.tree import DecisionTreeClassifier

import itertools

NUM_OF_METASTASES = 11
K = 10
ALPHA = 0.3

COLUMN_NAMES_SPACE = ['ADR - Adrenals', 'BON - Bones', 'BRA - Brain', 'HEP - Hepatic', 'LYM - Lymphnodes',
                      'MAR - BoneMarrow', 'OTH - Other', 'PER - Peritoneum', 'PLE - Pleura', 'PUL - Pulmonary',
                      'SKI - Skin']


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
    return pd.DataFrame(dummies['אבחנה-Location of distal metastases'], columns=['אבחנה-Location of distal metastases'])
    # return dummies['אבחנה-Location of distal metastases']


def predicting_metastases_v1(X_train, X_test, y_train, col_names):
    random_forest = ensemble.RandomForestClassifier(max_depth=10, random_state=42, class_weight="balanced")
    classifier = OneVsRestClassifier(estimator=random_forest)
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)

    return indicator_matrix_to_lists(pred, col_names)


def preprocess_y(y_file):
    dfy_train = parse_df_labels(pd.read_csv(y_file, keep_default_na=False))
    enc = Encode_Multi_Hot()
    dfy_train_vals = dfy_train["vals"]
    enc.fit(dfy_train_vals)  # get all possible labels
    dfy_train_multi_hot = np.array([enc.enc(val) for val in dfy_train_vals])
    col_names = [enc.ind_to_label[i] for i in range(len(enc.ind_to_label))]
    return dfy_train_multi_hot, col_names


def load_files_to_array(train_x_file, test_x_file, train_y_file):
    # load files
    dfX_train = preprocessor(load_data(train_x_file))
    dfX_test = preprocessor(load_data(train_y_file))

    # get labels

    dfy_train_multi_hot, col_names = preprocess_y(test_x_file)
    X_train, y_train = dfX_train.to_numpy(), dfy_train_multi_hot.astype(int)
    X_test = dfX_test.to_numpy()
    return X_train, y_train, X_test, col_names


def predicting_tumer_size_v1(X_train, X_test, y_train, y_test):
    lasso = Lasso(ALPHA)
    lasso.fit(X_train, y_train)


def run_predict_q1(X_train_file, y_train_file, X_test_file):
    X_train, y_train, X_test, col_names = load_files_to_array(X_train_file, y_train_file, X_test_file)

    # Q1
    return predicting_metastases_v1(X_train, X_test, y_train, col_names)


if __name__ == '__main__':
    X_train, y_train, X_test, col_names = load_files_to_array(sys.argv)
    #  TODO: remove - its for debugging
    # train_after_parse = indicator_matrix_to_lists(y_test, col_names)
    # pd.DataFrame(train_after_parse).to_csv("tree_pred.csv", index=False)

    # Q1
    predicting_metastases_v1(X_train, X_test, y_train, col_names)

    # Q2
    # predicting_tumer_size_v1(X_train, X_test, y_train, y_test)
