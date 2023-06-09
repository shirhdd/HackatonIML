import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from ..main import parse_df_labels

TEST_FILA_PATH = "../tests_sets/test1_label_1.csv"


def predicting_tumer_size(X_train, X_test, y_train):
    gold_fn = TEST_FILA_PATH
    gold_labels = parse_df_labels(pd.read_csv(gold_fn, keep_default_na=False, dtype={'אבחנה-Tumor size': str}))
    gold_vals = gold_labels["vals"]

    alphas = np.linspace(2600, 2800, 60)
    loss = np.zeros(len(alphas))
    for i, a in enumerate(alphas):
        estimator = Ridge(alpha=a)
        estimator.fit(X_train, y_train)
        pred = estimator.predict(X_test)
        np.where(pred < 0, 0, pred)
        loss[i] = mean_squared_error(y_true=gold_vals, y_pred=pred)

    print("argmin: ", alphas[np.argmin(loss)])
    estimator = Ridge(alpha=alphas[np.argmin(loss)])
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)

    result_eval_regression(y_pred=y_pred, y_true=np.array(gold_vals))


def result_eval_classification(y_pred, y_true):
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


def result_eval_regression(y_pred, y_true):
    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    print("mean square error ", mse)

    # Trivial performance for reference
    trivial_mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    print("trivial mse: ", trivial_mse)
