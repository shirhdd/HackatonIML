import numpy as np
import pandas as pd
import sklearn
from sklearn import decomposition
from sklearn import datasets
import Preprocessor
if __name__ == '__main__':
    x = Preprocessor.main('given_sets\\train.feats.csv').to_numpy()
    y = pd.read_csv('given_sets\\train.labels.0.csv').to_numpy()
    label_names = np.unique(y)
    label_counts = np.array(np.unique(y, return_counts=True)).T
    sorted_label_counts = label_counts[label_counts[:, 1].argsort()[::-1]]
    pca = decomposition.PCA(n_components=2, whiten=True)
    pca.fit(x)
    X_pca = pca.transform(x)
    target_ids = range(len(label_names))

    from matplotlib import pyplot as plt

    # plt.figure(figsize=(6, 5))
    # for i, label in zip(target_ids, label_names):
    #     plt.scatter(x=pd.DataFrame(X_pca).iloc[y == label_names[i], 0],
    #                 y=pd.DataFrame(X_pca).iloc[y == label_names[i], 1], label=label)
    # # plt.legend()
    # plt.show()
    #
    # plt.figure(figsize=(6, 5))
    # for i, label in zip(target_ids, label_names):
    #     if label == '[]':
    #         continue
    #     plt.scatter(x=pd.DataFrame(X_pca).iloc[y == label_names[i], 0],
    #                 y=pd.DataFrame(X_pca).iloc[y == label_names[i], 1], label=label)
    # # plt.legend()
    # plt.show()

    plt.figure(figsize=(10, 5))
    for i, label in zip(range(7), label_names):
        if label == '[]':
            continue
        plt.scatter(x=pd.DataFrame(X_pca).iloc[y == label_names[i], 0],
                    y=pd.DataFrame(X_pca).iloc[y == label_names[i], 1], label=label)
    # plt.legend()
    plt.show()
