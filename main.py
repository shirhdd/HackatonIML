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
	return dummies['אבחנה-Location of distal metastases']


def predicting_metastases_v1(X_train, X_test, y_train, col_names):
	random_forest = ensemble.RandomForestClassifier(max_depth=10, random_state=42, class_weight="balanced")
	classifier = OneVsRestClassifier(estimator=random_forest)
	classifier.fit(X_train, y_train)
	pred = classifier.predict(X_test)

	pred_df = indicator_matrix_to_lists(pred, col_names)
	pred_df.to_csv("tree_pred.csv", index=False)


def preprocess_y(y_file):
	dfy_train = parse_df_labels(pd.read_csv(y_file, keep_default_na=False))
	enc = Encode_Multi_Hot()
	dfy_train_vals = dfy_train["vals"]
	enc.fit(dfy_train_vals)  # get all possible labels
	dfy_train_multi_hot = np.array([enc.enc(val) for val in dfy_train_vals])
	col_names = [enc.ind_to_label[i] for i in range(len(enc.ind_to_label))]
	return dfy_train_multi_hot, col_names


def load_files_to_array(argv):
	# load files

	dfX_train = preprocessor(load_data(argv[1]))
	dfX_test = preprocessor(load_data(argv[3]))
	y_test = pd.read_csv(argv[4]).to_numpy()

	# get labels

	dfy_train_multi_hot, col_names = preprocess_y(argv[2])
	X_train, y_train = dfX_train.to_numpy(), dfy_train_multi_hot.astype(int)
	X_test = dfX_test.to_numpy()
	return X_train, y_train, X_test, y_test, col_names


def predicting_tumer_size_v1(X_train, X_test, y_train, y_test):
	lasso = Lasso(ALPHA)
	lasso.fit(X_train, y_train)


# print("Score on q2 very basic: ", lasso.score(X_test, y_test))


if __name__ == '__main__':
	X_train, y_train, X_test, y_test, col_names = load_files_to_array(sys.argv)
	#  TODO: remove - its for debugging
	# train_after_parse = indicator_matrix_to_lists(y_test, col_names)
	# pd.DataFrame(train_after_parse).to_csv("tree_pred.csv", index=False)

	# Q1
	predicting_metastases_v1(X_train, X_test, y_train, col_names)

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


# import pandas as pd
# import numpy as np
# import random
# import sys
# import os
# from sklearn import ensemble
# from sklearn.impute import SimpleImputer
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.multiclass import OneVsRestClassifier
#
# LYMPHOVASCULAR_INVASION = 'אבחנה-Ivi -Lymphovascular invasion'
#
# STAGE = 'אבחנה-Stage'
#
# N_LYMPH_NODES_MARK_TNM_ = 'אבחנה-N -lymph nodes mark (TNM)'
#
# T_TUMOR_MARK_TNM_ = 'אבחנה-T -Tumor mark (TNM)'
#
# LYMPHATIC_PENETRATION = 'אבחנה-Lymphatic penetration'
#
# BASIC_STAGE_NULL = 'אבחנה-Basic stage_Null'
#
# LABELS_TITLE = "אבחנה-Location of distal metastases"
#
# LOCATION_TITLE = "אבחנה-Location of distal metastases"
# KI67_TITLE = 'אבחנה-KI67 protein'
# LYMPHOVASCULAR_INVASION_TITLE = 'אבחנה-Ivi -Lymphovascular invasion'
# HISTOLOGICAL_TITLE = 'אבחנה-Histological diagnosis'
# HISTOLOGICAL_DEGREE_TITLE = 'אבחנה-Histopatological degree'
# ID_TITLE = 'id-hushed_internalpatientid'
# HER2_TITLE = 'אבחנה-Her2'
#
#
# ######################################### Helper Functions For Pre-Proceesing #########################################
# def add_dummies(title, X):
#     dummies = pd.get_dummies(X[title], prefix=title, dummy_na=False)
#     X = pd.concat([X, dummies], axis=1)
#     return X
#
#
# def degree_to_numeric(CancerDataFrame):
#     degree_list = sorted(list(set(CancerDataFrame[HISTOLOGICAL_DEGREE_TITLE])))
#     degree_dict = dict()
#     for index, degree in enumerate(degree_list):
#         if index < 4:
#             degree_dict[degree] = index + 1
#         else:
#             degree_dict[degree] = None
#     return degree_dict
#
#
# def id_to_num_of_visit(X_train):
#     keys, values = np.unique(X_train[ID_TITLE], return_counts=True)
#     return dict(zip(keys, values))
#
#
# def find_histological(CancerDataFrame):
#     hist_set = set(CancerDataFrame[HISTOLOGICAL_TITLE])
#     for diagnosis in hist_set:
#         lst = CancerDataFrame[CancerDataFrame[HISTOLOGICAL_TITLE] == diagnosis].index.tolist()
#         not_non = list(CancerDataFrame[HISTOLOGICAL_DEGREE_TITLE][lst].dropna())
#         if not not_non:
#             continue
#         mean = np.round(np.mean(not_non))
#         CancerDataFrame[HISTOLOGICAL_DEGREE_TITLE][lst] = CancerDataFrame[HISTOLOGICAL_DEGREE_TITLE][lst].fillna(mean)
#
#
# def Her2_helper(current_cell):
#     current_cell = str(current_cell)
#     # positive -> 3
#     if ("POS" in current_cell or "pos" in current_cell or "Pos" in current_cell or "+" in current_cell or
#             "3" in current_cell or "jhuch" in current_cell):
#         return 3
#     # negative -> 0
#     elif ("NEG" in current_cell or "Neg" in current_cell or "neg" in current_cell or "-" in current_cell or
#           "akhah" in current_cell):
#         return 0
#     elif ("Equivocal" in current_cell or "equivocal" in current_cell or "2" in current_cell):
#         return 2
#     elif ("1" in current_cell):
#         return 1
#     else:
#         return None
#
#
# def Ivi_helper(current_cell):
#     current_cell = str(current_cell)
#     zero_list = ["-", "neg", "Neg", "NEG", "no", "No", "NO", "Not"]
#     one_list = ["+", "pos", "Pos", "POS", "yes", "Yes", "YES", "extensive", "Extensive", "EXTENSIVE",
#                 "MICROPAPILLARY VARIANT"]
#     for item in zero_list:
#         if item in current_cell:
#             return 0
#     for item in one_list:
#         if item in current_cell:
#             return 1
#     return None
#
#
# def Her2(X_train):
#     X_train[HER2_TITLE] = X_train[HER2_TITLE].apply(Her2_helper)
#     return X_train[HER2_TITLE]
#
#
# def Ivi(X_train):
#     X_train[LYMPHOVASCULAR_INVASION_TITLE] = X_train[LYMPHOVASCULAR_INVASION_TITLE].apply(Ivi_helper)
#     return X_train[LYMPHOVASCULAR_INVASION_TITLE]
#
#
# def handle_lymphatic_penetration(x):
#     if x == 'Null':
#         return None
#     elif x[0:2] == 'LI':
#         return 1.5
#     else:
#         return int(x[1])
#
#
# def lymph_nodes_mark(x):
#     dict_N = {'Null': None, 'N0': 0, 'N1a': 1, 'N1': 1, 'N1b': 1, 'N1c': 1.5, 'N1d': 1.5
#         , 'N2a': 2, 'N2': 2, 'N2b': 2, 'N2c': 2.5, 'N2d': 2.5
#         , 'N3a': 3, 'N3': 3, 'N3b': 3, 'N3c': 3.5, 'N3d': 3.5, 'N4': 4, 'Nx': None}
#     if x in dict_N.keys():
#         return dict_N[x]
#     else:
#         return None
#
#
# def lymph_tumor_mark(x):
#     dict = {'Null': None, 'T0': 0, 'Tis': 0.5, 'T1': 1.5, 'T1mi': 1, 'T1a': 1.2, 'T1b': 1.4, 'T1c': 1.6, 'T1d': 1.8
#         , 'T2': 2.5, 'T2mi': 2, 'T2a': 2.2, 'T2b': 2.4, 'T2c': 2.6, 'T2d': 2.8,
#             'T3': 3.5, 'T3mi': 3, 'T3a': 3.2, 'T3b': 3.4, 'T3c': 3.6, 'T3d': 3.8,
#             'T4': 4.5, 'T4mi': 4, 'T4a': 4.2, 'T4b': 4.4, 'T4c': 4.6, 'T4d': 4.8}
#     if x in dict.keys():
#         return dict[x]
#     else:
#         return None
#
#
# def cancer_stage(x):
#     dict = {'LA': 0.2, 'Stage0': 0, 'Stage0is': 0.5, 'Stage0a': 0.2, 'Stage0b': 0.4, 'Stage0c': 0.6, 'Stage0d': 0.8,
#             'Stage1': 1.5, 'Stage1a': 1.2, 'Stage1b': 1.4, 'Stage1c': 1.6, 'Stage1d': 1.8
#         , 'Stage2': 2.5, 'Stage2mi': 2, 'Stage2a': 2.2, 'Stage2b': 2.4, 'Stage2c': 2.6, 'Stage2d': 2.8,
#             'Stage3': 3.5, 'Stage3mi': 3, 'Stage3a': 3.2, 'Stage3b': 3.4, 'Stage3c': 3.6, 'Stage3d': 3.8,
#             'Stage4': 4.5, 'Stage4a': 4.2, 'Stage4b': 4.4, 'Stage4c': 4.6, 'Stage4d': 4.8}
#     if x in dict.keys():
#         return dict[x]
#     else:
#         return None
#
#
# def KI67(X_train):
#     X_train[KI67_TITLE] = X_train[KI67_TITLE].apply(KI67_helper)
#     X_train[KI67_TITLE] = np.where(X_train[KI67_TITLE] > 1, None, X_train[KI67_TITLE])
#     return X_train[KI67_TITLE]
#
#
# def KI67_helper(current_cell):
#     current_cell = str(current_cell)
#     percent_index = current_cell.find("%")
#     if percent_index != -1:
#         current_cell = current_cell[:percent_index]
#     current_cell = current_cell.replace('%', '')
#     current_cell = current_cell.replace('>', '')
#     current_cell = current_cell.replace('<', '')
#     current_cell = current_cell.replace('+', '')
#
#     # if only number -> put it
#     if current_cell.isnumeric():
#         return float(current_cell) / 100
#     # if has "score"
#     elif "Score" in current_cell or "score" in current_cell:
#
#         if "1" in current_cell or current_cell.count("I") == 1 and current_cell.count("V") == 0:
#             return 2.0 / 100
#         elif "2" in current_cell or current_cell.count("I") == 2 and current_cell.count("V") == 0:
#             return 12.5 / 100
#         elif "3" in current_cell or current_cell.count("I") == 3 and current_cell.count("V") == 0:
#             return 20 / 100
#         elif "4" in current_cell or "IV" in current_cell:
#             return 50 / 100
#         else:
#             return None
#
#     elif "-" in current_cell:
#         splitted_array = current_cell.split("-")
#         if len(splitted_array[0]) == 1:
#             first_number = splitted_array[0]
#         else:
#             first_number = splitted_array[0][-2:]
#         if len(splitted_array[1]) == 1:
#             second_number = splitted_array[1]
#         else:
#             second_number = splitted_array[1][:2]
#
#         if first_number.isnumeric() and second_number.isnumeric():
#             return ((int(first_number) + int(second_number)) / 2.0) / 100.0
#         else:
#             return None
#
#
# def er_pr_handler(X_train, title):
#     X_train[title] = X_train[title].apply(er_helper)
#     X_train[title] = np.where(X_train[title] > 1, None, X_train[title])
#     X_train[title] = np.where(X_train[title] < 0, None, X_train[title])
#     return X_train[title]
#
#
# def split_for_er(current_cell):
#     current_cell = str(current_cell)
#     zero_list = ["-", "neg", "Neg", "NEG", "no", "No", "NO", "Not", "nge", "akhkh", "שלילי"]
#     one_list = ["+", "pos", "Pos", "POS", "yes", "Yes", "YES", "po", "Po", "jhuch", "חיובי"]
#     for item in zero_list:
#         if item in current_cell:
#             return 0
#     for item in one_list:
#         if item in current_cell:
#             return 1
#     return np.nan
#
#
# def er_helper(current_cell):
#     current_cell = str(current_cell)
#     percent_index = current_cell.find("%")
#     if percent_index != -1:
#         for i in range(3, 0, -1):
#             check = current_cell[percent_index - i:percent_index]
#             if check.isnumeric():
#                 return float(check) / 100
#
#     current_cell = current_cell.replace('%', '')
#     current_cell = current_cell.replace('>', '')
#     current_cell = current_cell.replace('<', '')
#     current_cell = current_cell.replace('^', '')
#     # if only number -> put it
#     if current_cell.isnumeric():
#         return float(current_cell) / 100
#     return split_for_er(current_cell)
#
#
# ######################################### Splitting the Data #########################################
# def split_data(train_features, train_labels):
#     X = pd.read_csv(train_features)
#     X_names = X.columns
#     y = pd.read_csv(train_labels)
#     y_names = y.columns
#     X, y = preprocess(X, y, True)
#     save_y = y.copy()
#     y_array = y.to_numpy().ravel()
#     y_array = list(map(eval, y_array))
#     y_array = list(map(sorted, y_array))
#     y_updated = list()
#     for index, label in enumerate(y_array):
#         if label and len(label) > 1:
#             for prediction_item in label:
#                 y_updated.append([prediction_item])
#         else:
#             y_updated.append(y_array[index])
#     unique_labels = np.unique(y_updated)
#     unique_labels = unique_labels[1:]
#     for label in unique_labels:
#         y[label] = 0
#     for index, classifcations in enumerate(y_array):
#         if classifcations:
#             for element in classifcations:
#                 y.at[index, element] = 1
#     y = y.drop(LABELS_TITLE, 1)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0, shuffle=True)
#     return X_train, y_train, X_test, y_test, save_y[LABELS_TITLE][y_test.index.tolist()]
#
#
# def preprocess(X, y, flag=False):
#     dummies_list = [' Hospital', 'אבחנה-M -metastases mark (TNM)', 'אבחנה-Basic stage',
#                     'אבחנה-Margin Type', 'אבחנה-Side']
#     drop_list = ['אבחנה-Surgery date1', 'אבחנה-Surgery date2', 'אבחנה-Surgery date3', 'אבחנה-Surgery name1',
#                  'אבחנה-Surgery name2', 'אבחנה-Surgery name3', 'surgery before or after-Actual activity',
#                  'surgery before or after-Activity date', 'אבחנה-Diagnosis date', 'אבחנה-Tumor depth', ' Form Name']
#     for dummy in dummies_list:
#         X = add_dummies(dummy, X)
#         drop_list.append(dummy)
#
#     drop_list.append('User Name')
#     drop_list.append(BASIC_STAGE_NULL)
#     degree_dict = degree_to_numeric(X)
#     X = X.replace({HISTOLOGICAL_DEGREE_TITLE: degree_dict})
#     find_histological(X)
#     drop_list.append(HISTOLOGICAL_TITLE)
#     X[LYMPHATIC_PENETRATION] = \
#         X[LYMPHATIC_PENETRATION].apply(handle_lymphatic_penetration)
#     X[T_TUMOR_MARK_TNM_] = \
#         X[T_TUMOR_MARK_TNM_].apply(lymph_tumor_mark)
#     X[HER2_TITLE] = Her2(X)
#     X[N_LYMPH_NODES_MARK_TNM_] = \
#         X[N_LYMPH_NODES_MARK_TNM_].apply(lymph_nodes_mark)
#     X[STAGE] = \
#         X[STAGE].apply(cancer_stage)
#     X[LYMPHOVASCULAR_INVASION] = Ivi(X)
#     X[KI67_TITLE] = KI67(X)
#     X['אבחנה-er'] = er_pr_handler(X, 'אבחנה-er')
#     X['אבחנה-pr'] = er_pr_handler(X, 'אבחנה-pr')
#     X = X.drop(drop_list, axis=1)
#     X = X.drop([ID_TITLE], axis=1)
#
#     if flag:
#         full_data = pd.concat([X, y], axis=1).drop_duplicates()
#         X = full_data.iloc[:, :-1]
#         X.reset_index(inplace=True, drop=True)
#         y = full_data.iloc[:, -1:]
#         y.reset_index(inplace=True, drop=True)
#
#         for index, row in X.iterrows():
#             count = row.dropna().shape[0] / row.shape[0]
#             if count < 0.6:
#                 X = X.drop([index])
#                 y = y.drop([index])
#     a = SimpleImputer(missing_values=np.nan, strategy='mean')
#     X = pd.DataFrame(a.fit_transform(X.values), columns=X.columns)
#     X.reset_index(inplace=True, drop=True)
#     if flag:
#         y.reset_index(inplace=True, drop=True)
#     return X, y
#
#
# def Part1Train(train_features, train_labels, test_features):
#     X_train, y_train, X_test, y_test, save_y = split_data()
#     save_y.to_csv("golden.csv", index_label=False)
#     diag_names = y_train.columns
#     X_to_predict = pd.read_csv(test_features)
#     X_to_predict, _ = preprocess(X_to_predict, [])
#     # X_to_predict['אבחנה-M -metastases mark (TNM)_M1a'] = 0
#     # using binary relevance
#     X_train = X_train.to_numpy()
#     y_train = y_train.to_numpy()
#
#     from sklearn.metrics import accuracy_score
#     from sklearn.multiclass import OneVsRestClassifier
#
#     random_forest = ensemble.RandomForestClassifier(random_state=42, class_weight="balanced")
#     classifier = OneVsRestClassifier(estimator=random_forest)
#     classifier.fit(X_train, y_train)
#     predictions = classifier.predict(X_test)
#     # predictions = classifier.predict(X_to_predict)
#     new_predictions = []
#     for row in predictions:
#         index_list = np.nonzero(row)
#         if 1 not in row:
#             new_predictions.append([])
#         else:
#             new_predictions.append(list(np.take(diag_names, index_list, axis = 0)))
#     end = pd.DataFrame()
#     end["אבחנה-Location of distal metastases"] = new_predictions
#     end.to_csv("Pred1.csv", index_label=False)
#
# # def Part2Train(train_features, train_labels, test_features):
# #     X = pd.read_csv(train_features)
# #     y = pd.read_csv(train_labels)
# #     X, y = preprocess(X, y, True)
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0, shuffle=True)
# #     y_test.to_csv("golden-2.csv", index_label=False)
# #
# #     model = GradientBoostingRegressor()
# #     model.fit(X_train, y_train)
# #     y_pred = model.predict(X_test)
# #     end = pd.DataFrame()
# #     end['אבחנה-Tumor size'] = y_pred
# #     end.to_csv("Pred2.csv", index_label=False)
# #     # print(model)
# #     # print("\tR2 score:", mean_squared_error(y_test, y_pred))
#
# def Part1(train_features, train_labels, test_features):
#     X_train, y_train, X_test, y_test, save_x = split_data(train_features, train_labels)
#     diag_names = y_train.columns
#     X_to_predict = pd.read_csv(test_features)
#     X_to_predict, _ = preprocess(X_to_predict, [])
#     X_to_predict['אבחנה-M -metastases mark (TNM)_M1a'] = 0
#
#     random_forest = ensemble.RandomForestClassifier(random_state=42, class_weight="balanced")
#     classifier = OneVsRestClassifier(estimator=random_forest)
#     classifier.fit(X_train, y_train)
#     predictions = classifier.predict(X_to_predict)
#     new_predictions = []
#     for row in predictions:
#         index_list = np.nonzero(row)
#         if 1 not in row:
#             new_predictions.append([])
#         else:
#             new_predictions.append(list(np.take(diag_names, index_list, axis=0)))
#     end = pd.DataFrame()
#     end["אבחנה-Location of distal metastases"] = new_predictions
#     end.to_csv("prediction_0_classification.csv", index=False)
#
#
# def Part2(train_features, train_labels, test_features):
#     X = pd.read_csv(train_features)
#     y = pd.read_csv(train_labels)
#     X, y = preprocess(X, y, True)
#     X_to_predict = pd.read_csv(test_features)
#     X_to_predict, _ = preprocess(X_to_predict, [])
#     X_to_predict['אבחנה-M -metastases mark (TNM)_M1a'] = 0
#
#     model = GradientBoostingRegressor()
#     model.fit(X, y)
#     y_pred = model.predict(X_to_predict)
#     end = pd.DataFrame()
#     end['אבחנה-Tumor size'] = y_pred
#     end.to_csv("prediction_1_regression.csv", index=False)
#
# if __name__ == '__main__':
#     # Usage:
#     # <program_name> <seed> <1/2>  <train features file name> <train labels file name> <predict set>
#     # Please Use seed 0 to get the same results we got :)
#     if not len(sys.argv) == 6:
#         print("Usage: <program_name> <seed> <0/1>  <train features file name> <train labels file name> <predict set>",
#               file=sys.stderr)
#         exit(1)
#     if not sys.argv[1].isnumeric() and int(sys.argv[1]) >= 0:
#         print("Usage: Seed must be a non negative integer", file=sys.stderr)
#         exit(1)
#     random.seed(sys.argv[1])
#     if sys.argv[2] != "0" and sys.argv[2] != "1":
#         print("Usage: this argument need to be in {1,2} only \n 1  - to run the first task "
#               "\n 2 - ti run the second task", file=sys.stderr)
#         exit(1)
#     for i in range(3, 6):
#         if not os.path.exists(sys.argv[i]):
#             print("Usage: invalid path please enter an existing file", file=sys.stderr)
#             exit(1)
#
#     if sys.argv[2] == "0":
#         Part1(sys.argv[3], sys.argv[4], sys.argv[5])
#     else:
#         Part2(sys.argv[3], sys.argv[4], sys.argv[5])
