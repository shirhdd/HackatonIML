import sys
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import Lasso
from Preprocessor import preprocessor, preprocess_labels_q1, load_data
from sklearn.tree import DecisionTreeClassifier

import itertools


NUM_OF_METASTASES = 11
K = 10
ALPHA = 0.3
COLUMN_NAMES = ['ADR-Adrenals', 'BON-Bones', 'BRA-Brain', 'HEP-Hepatic', 'LYM-Lymphnodes', 'MAR-BoneMarrow',
                'OTH-Other', 'PER-Peritoneum', 'PLE-Pleura', 'PUL-Pulmonary',  'SKI-Skin']

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
    assert(len(df.columns) == 1)
    resp = df.columns[0]
    ls = [eval(val) for val in df[resp]]
    ret_dict = {"resp": resp, "vals": ls}
    return ret_dict



def indicator_matrix_to_lists(dummies):
    true_columns = []
    dummies = pd.DataFrame(dummies, columns=COLUMN_NAMES)
    for index, row in dummies.iterrows():
        true_values = row[row == 1].index.tolist()
        true_columns.append(true_values)
    # df = pd.DataFrame(true_columns, columns="lable")
    dummies['אבחנה-Location of distal metastases'] = true_columns
    return dummies['אבחנה-Location of distal metastases']
    # list_predict = []
    # for row in y:
    #     index_list = np.nonzero(row)
    #     if 1 not in row:
    #         list_predict.append([])
    #     else:
    #         list_predict.append(list(np.take(COLUMN_NAMES, index_list, axis=0)))
    # nList = np.array()
    # return pd.DataFrame(list_predict)

def predicting_metastases_v1(X_train, X_test, y_train, y_test):
    tree = DecisionTreeClassifier()
    multiclass = MultiOutputClassifier(tree, NUM_OF_METASTASES)
    multiclass.fit(X_train, y_train)
    pred = multiclass.predict(X_test)

    pred_df = indicator_matrix_to_lists(pred)

    pd.DataFrame(pred_df).to_csv("tree_pred.csv", index=False)


def predicting_tumer_size_v1(X_train, X_test, y_train, y_test):
    lasso = Lasso(ALPHA)
    lasso.fit(X_train, y_train)
    print("Score on q2 very basic: ", lasso.score(X_test, y_test))


if __name__ == '__main__':
    dfX_train = preprocessor(load_data(sys.argv[1]))
    # dfy_train = preprocess_labels_q1(sys.argv[2])
    dfX_test = preprocessor(load_data(sys.argv[3]))
    # dfy_test = preprocess_labels_q1(sys.argv[4])

    dfy_train = parse_df_labels(pd.read_csv(sys.argv[2], keep_default_na=False))
    dfy_test = parse_df_labels(pd.read_csv(sys.argv[4], keep_default_na=False))
    enc = Encode_Multi_Hot()
    dfy_train_vals = dfy_train["vals"]
    dfy_test_vals = dfy_test["vals"]
    enc.fit(dfy_train_vals + dfy_test_vals)
    dfy_train_multi_hot = np.array([enc.enc(val) for val in dfy_train_vals])
    dfy_test_multi_hot = np.array([enc.enc(val) for val in dfy_test_vals])

    # dfy_test = dfy_test.reindex(columns=COLUMN_NAMES, fill_value=0)
    # dfy_train = dfy_train.reindex(columns=COLUMN_NAMES, fill_value=0)
    X_train, y_train = dfy_train_multi_hot, dfy_test_multi_hot
    X_test, y_test = dfX_test.to_numpy(), dfy_test.to_numpy()

    pd.DataFrame(y_test).to_csv("tree_test.csv")

    # Q1
    predicting_metastases_v1(X_train, X_test, y_train, y_test)

    # Q2
    # predicting_tumer_size_v1(X_train, X_test, y_train, y_test)




# def result_eval(y_pred, y_true):
#     y_pred = y_pred.reshape(y_pred.size)
#     y_true = y_true.reshape(y_true.size)
#     wrong = np.sum(y_pred != y_true)
#     pred_positive_index = np.where(y_pred != '[]')
#     false_positive = np.sum(y_pred[pred_positive_index] != y_true[pred_positive_index])
#     pred_negative_index = np.where(y_pred == '[]')
#     print("number of wrong classifications", wrong)
#     print("number of correct classification ", y_pred.size - wrong)
#     print("number of false positive", false_positive)
#     print("number of false negative ", wrong-false_positive)
#     print("number of true positive ", )
#     print("predicted negative ", len(pred_negative_index[0]))

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


# def predicting_metastases_v1(X_train, X_test, y_train, y_test):
#     tree = DecisionTreeClassifier()
#     tree.fit(X_train, y_train)
#     pred = tree.predict(X_test)
#     result_eval(pred, y_test)
#     pd.DataFrame(pred, columns=["אבחנה-Location of distal metastases"]).to_csv("tree_pred.csv", index=False)