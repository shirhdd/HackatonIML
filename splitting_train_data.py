import numpy as np
import pandas as pd
import sklearn

TRAIN_FILE_PATH = "given_sets/train.feats.csv"
TEST_FILE_PATH = "given_sets/test.feats.csv"
LABEL_0_FILE_PATH = "original_train_labels_0_DO_NOT_USE.csv"
LABEL_1_FILE_PATH = "original_train_labels_1_DO_NOT_USE.csv"


def load_train_data():
    df = pd.read_csv(TRAIN_FILE_PATH, dtype=str)
    return df


def split_data(df):
    patients = df['id-hushed_internalpatientid'].unique()
    patients = np.random.permutation(patients)
    num_of_patients = len(patients)
    train, test1, test2, test3, test4, test5 = np.array(
        np.ceil([0.4 * num_of_patients,
                 0.5 * num_of_patients,
                 0.6 * num_of_patients,
                 0.7 * num_of_patients,
                 0.8 * num_of_patients,
                 0.9 * num_of_patients])).astype(int)
    train, test1, test2, test3, test4, test5, test6 = np.split(patients,
                                                               [train, test1,
                                                                test2, test3,
                                                                test4, test5])
    df[df['id-hushed_internalpatientid'].isin(train)].to_csv('train_df.csv')
    df[df['id-hushed_internalpatientid'].isin(test1)].to_csv('test1.csv')
    df[df['id-hushed_internalpatientid'].isin(test2)].to_csv('test2.csv')
    df[df['id-hushed_internalpatientid'].isin(test3)].to_csv('test3.csv')
    df[df['id-hushed_internalpatientid'].isin(test4)].to_csv('test4.csv')
    df[df['id-hushed_internalpatientid'].isin(test5)].to_csv('test5.csv')
    df[df['id-hushed_internalpatientid'].isin(test6)].to_csv('test6.csv')


if __name__ == '__main__':
    np.random.seed(0)
    df = load_train_data()
    split_data(df)
