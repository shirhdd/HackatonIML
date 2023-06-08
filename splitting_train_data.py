import numpy as np
import pandas as pd
import sklearn
from sklearn import decomposition
from sklearn import datasets
import Preprocessor

TRAIN_FILE_PATH = "given_sets\\train.feats.csv"
LABEL_0_FILE_PATH = "given_sets\\train.labels.0.csv"
LABEL_1_FILE_PATH = "given_sets\\train.labels.1.csv"


def load_train_data():
    df_train = pd.read_csv(TRAIN_FILE_PATH, dtype=str)
    label_0_train = pd.read_csv(LABEL_0_FILE_PATH)
    label_1_train = pd.read_csv(LABEL_1_FILE_PATH)
    con = df_train.join([label_0_train, label_1_train])
    return con


def split_data(df):
    all_patients = df['id-hushed_internalpatientid'].unique()
    ill = df.loc[df['אבחנה-Location of distal metastases'] != '[]', 'id-hushed_internalpatientid'].unique()
    healthy = df.loc[df['אבחנה-Location of distal metastases'] == '[]', 'id-hushed_internalpatientid'].unique()
    ill = np.random.permutation(ill)
    num_of_ill_patients = len(ill)
    train, test1, test2, test3, test4, test5 = np.array(
        np.ceil([0.4 * num_of_ill_patients,
                 0.5 * num_of_ill_patients,
                 0.6 * num_of_ill_patients,
                 0.7 * num_of_ill_patients,
                 0.8 * num_of_ill_patients,
                 0.9 * num_of_ill_patients])).astype(int)
    train_ill, test1_ill, test2_ill, test3_ill, test4_ill, test5_ill, test6_ill = np.split(ill,
                                                                                           [train, test1,
                                                                                            test2, test3,
                                                                                            test4, test5])
    healthy = np.random.permutation(healthy)
    num_of_healthy_patients = len(healthy)
    train, test1, test2, test3, test4, test5 = np.array(
        np.ceil([0.4 * num_of_healthy_patients,
                 0.5 * num_of_healthy_patients,
                 0.6 * num_of_healthy_patients,
                 0.7 * num_of_healthy_patients,
                 0.8 * num_of_healthy_patients,
                 0.9 * num_of_healthy_patients])).astype(int)
    train_healthy, test1_healthy, test2_healthy, test3_healthy, test4_healthy, test5_healthy, test6_healthy = np.split(
        healthy,
        [train, test1,
         test2, test3,
         test4, test5])
    train = df[
        (df['id-hushed_internalpatientid'].isin(train_ill)) | (df['id-hushed_internalpatientid'].isin(train_healthy))]
    test1 = df[
        (df['id-hushed_internalpatientid'].isin(test1_ill)) | (df['id-hushed_internalpatientid'].isin(test1_healthy))]
    test2 = df[(df['id-hushed_internalpatientid'].isin(test2_ill)) | (
        df['id-hushed_internalpatientid'].isin(test2_healthy))]
    test3 = df[(df['id-hushed_internalpatientid'].isin(test3_ill)) | (
        df['id-hushed_internalpatientid'].isin(test3_healthy))]
    test4 = df[(df['id-hushed_internalpatientid'].isin(test4_ill)) | (
        df['id-hushed_internalpatientid'].isin(test4_healthy))]
    test5 = df[(df['id-hushed_internalpatientid'].isin(test5_ill)) | (
        df['id-hushed_internalpatientid'].isin(test5_healthy))]
    test6 = df[(df['id-hushed_internalpatientid'].isin(test6_ill)) | (
        df['id-hushed_internalpatientid'].isin(test6_healthy))]

    train.iloc[:, :-2].to_csv('train_sets\\train.csv', index=False)
    train.iloc[:, -2].to_csv('train_sets\\train_label_0.csv', index=False)
    train.iloc[:, -1].to_csv('train_sets\\train_label_1.csv', index=False)

    test1.iloc[:, :-2].to_csv('tests_sets\\test1.csv', index=False)
    test1.iloc[:, -2].to_csv('tests_sets\\test1_label_0.csv', index=False)
    test1.iloc[:, -1].to_csv('tests_sets\\test1_label_1.csv', index=False)

    test2.iloc[:, :-2].to_csv('tests_sets\\test2.csv', index=False)
    test2.iloc[:, -2].to_csv('tests_sets\\test2_label_0.csv', index=False)
    test2.iloc[:, -1].to_csv('tests_sets\\test2_label_1.csv', index=False)

    test3.iloc[:, :-2].to_csv('tests_sets\\test3.csv', index=False)
    test3.iloc[:, -2].to_csv('tests_sets\\test3_label_0.csv', index=False)
    test3.iloc[:, -1].to_csv('tests_sets\\test3_label_1.csv', index=False)

    test4.iloc[:, :-2].to_csv('tests_sets\\test4.csv', index=False)
    test4.iloc[:, -2].to_csv('tests_sets\\test4_label_0.csv', index=False)
    test4.iloc[:, -1].to_csv('tests_sets\\test4_label_1.csv', index=False)

    test5.iloc[:, :-2].to_csv('tests_sets\\test5.csv', index=False)
    test5.iloc[:, -2].to_csv('tests_sets\\test5_label_0.csv', index=False)
    test5.iloc[:, -1].to_csv('tests_sets\\test5_label_1.csv', index=False)

    test6.iloc[:, :-2].to_csv('tests_sets\\test6.csv', index=False)
    test6.iloc[:, -2].to_csv('tests_sets\\test6_label_0.csv', index=False)
    test6.iloc[:, -1].to_csv('tests_sets\\test6_label_1.csv', index=False)


# ADR-Adrenals  BON-Bones  BRA-Brain  HEP-Hepatic  LYM-Lymphnodes  MAR-BoneMarrow  OTH-Other  PER-Peritoneum  PLE-Pleura  PUL-Pulmonary  SKI-Skin
if __name__ == '__main__':
    np.random.seed(0)
    # df = load_train_data()
    # split_data(df)

