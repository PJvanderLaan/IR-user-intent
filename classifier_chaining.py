import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.multioutput import ClassifierChain
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from util.pickle_helper import get_pickled_data, store_data_as_pickle
from sklearn.model_selection import cross_val_score

CACHED_DATA_FILENAME = 'fitted_chain'

def chaining_svm(X, Y, C=0.5, gamma='scale', max_iter=-1, folds=10):
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=0)
	# base_clf = SVC(C=C, kernel='rbf', gamma=gamma, max_iter=max_iter)
	base_clf = SVC()
	
	chain = get_pickled_data(f'testchainpickle')
	# chain = ClassifierChain(base_clf, cv=folds, order='random', random_state=0)
	chain.fit(X_train, Y_train)
	y_pred = chain.predict(X_test)
	print(f'accuracy: {accuracy_score(Y_test, y_pred)}')
	print(f'f1 score: {f1_score(Y_test, y_pred, average="macro")}')
	print(f'precision: {precision_score(Y_test, y_pred, average="macro")}')
	print(f'recall_score: {recall_score(Y_test, y_pred, average="macro")}')
	store_data_as_pickle(chain, f'testchainpickle')


	# chains = [ClassifierChain(base_clf, cv=folds, order='random', random_state=i) for i in range(10)]
	# for index, chain in enumerate(chains):
	# 	print(f'Fitting chain {index} ...')
	# 	chain.fit(X_train, Y_train)
	# 	print(f'Done fitting, storing as {CACHED_DATA_FILENAME}-{index}.pickle ...')
	# 	store_data_as_pickle(chains, f'{CACHED_DATA_FILENAME}-{index}')
	
