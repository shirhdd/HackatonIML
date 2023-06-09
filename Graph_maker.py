import numpy as np
import pandas as pd
from sklearn import decomposition
from submission.task_2.code.hackathon_code import Preprocessor
import plotly.express as px
from submission.task_2.code.hackathon_code import evaluate_part_0

TRAIN_DATA = 'given_sets\\train.feats.csv'
LABEL_DATA = 'given_sets\\train.labels.0.csv'


def make_pca():
    x = Preprocessor.load_and_preproc(TRAIN_DATA).to_numpy()
    y = pd.read_csv(LABEL_DATA).to_numpy()
    label_counts = np.array(np.unique(y, return_counts=True)).T
    sorted_label_counts = label_counts[label_counts[:, 1].argsort()[::-1]]
    label_names = sorted_label_counts[:, 0]
    pca = decomposition.PCA(n_components=2, whiten=True)
    pca.fit(x)
    X_pca = pca.transform(x)
    target_ids = range(len(label_names))

    from matplotlib import pyplot as plt

    plt.figure(figsize=(6, 5))
    for i, label in zip(target_ids, label_names):
        plt.scatter(x=pd.DataFrame(X_pca).iloc[y == label_names[i], 0],
                    y=pd.DataFrame(X_pca).iloc[y == label_names[i], 1], label=label)
    # plt.legend()
    plt.title('All the patients')
    plt.show()

    plt.figure(figsize=(6, 5))
    for i, label in zip(target_ids, label_names):
        if label == '[]':
            continue
        plt.scatter(x=pd.DataFrame(X_pca).iloc[y == label_names[i], 0],
                    y=pd.DataFrame(X_pca).iloc[y == label_names[i], 1], label=label)
    # plt.legend()
    plt.title('All the patients with Metastases')
    plt.show()

    plt.figure(figsize=(10, 5))
    for i, label in zip(range(7), label_names):
        if label == '[]':
            continue
        plt.scatter(x=pd.DataFrame(X_pca).iloc[y == label_names[i], 0],
                    y=pd.DataFrame(X_pca).iloc[y == label_names[i], 1], label=label)
    # plt.legend()

    plt.title('10 most common Metastases groups')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.scatter(x=pd.DataFrame(X_pca).iloc[y == '[]', 0],
                y=pd.DataFrame(X_pca).iloc[y == '[]', 1], label='[]')
    # plt.legend()
    plt.title('Patients with no Metastases')
    plt.show()


def histogram():
    y = pd.read_csv(LABEL_DATA).to_numpy()
    label_counts = np.array(np.unique(y, return_counts=True)).T
    sorted_label_counts = label_counts[label_counts[:, 1].argsort()[::-1]]
    label_counts = pd.DataFrame(sorted_label_counts, columns=['group', 'count'])
    label_counts = label_counts[label_counts['group'] != '[]']
    px.bar(label_counts, x='group', y='count', title="Histogram showing the number of patients of each group").show()

    # on specific labels
    df = pd.read_csv(LABEL_DATA, keep_default_na=False)
    gold_labels = evaluate_part_0.parse_df_labels(df)

    enc = evaluate_part_0.Encode_Multi_Hot()
    gold_vals = gold_labels["vals"]
    enc.fit(gold_vals)

    gold_multi_hot = np.array([enc.enc(val) for val in gold_vals])
    df = pd.DataFrame(gold_multi_hot, columns=enc.ind_to_label.values())
    df = df.sum(axis=0).reset_index(name='Count')
    df = df.sort_values(by=['Count'], ascending=False)
    px.bar(df, x='index', y='Count', title='Histogram showing the number of patients for each Metastases').show()

if __name__ == '__main__':
    # make_pca()
    histogram()
    pass

