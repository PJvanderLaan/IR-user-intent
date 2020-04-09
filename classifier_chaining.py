import numpy as np
import matplotlib.pyplot as plt
# from skmultilearn.problem_transform import ClassifierChain
from sklearn.multioutput import ClassifierChain
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from util.pickle_helper import get_pickled_data, store_data_as_pickle
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

CACHED_DATA_FILENAME = 'fitted_chain'

def chaining_svm(X, Y, C=0.5, max_iter=-1, folds=10):
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=0)

	# https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
	# https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
	# C_range = np.logspace(-2, 10, 13)
	# gamma_range = np.logspace(-9, 3, 13)
	# parameters = [
	#     {
	#         'classifier': [SVC()],
	#         'classifier__kernel': ['rbf'], 
	#         'classifier__gamma': gamma_range,
	#         'classifier__C': C_range,
	#     }
	# ]

	# # {
 # #        'classifier': [SVC()],
 # #        'classifier__kernel': ['linear'], 
 # #        'classifier__C': [1, 10, 100, 1000],
 # #    },

	# clf = GridSearchCV(ClassifierChain(), parameters, scoring='accuracy', verbose=10)
	# clf.fit(X_train, Y_train)
	# print(clf.best_params_, clf.best_score_)
	# store_data_as_pickle(clf, f'{CACHED_DATA_FILENAME}-gridsearchfitted')

	print('C \t accuracy \t f1 \t precision \t recall')
	for C in range(1, 10010, 10):
		base_clf = SVC(C=C, kernel='rbf', max_iter=max_iter)

		# chain = get_pickled_data(f'testchainpickle')
		chain = ClassifierChain(base_clf, cv=5, order='random', random_state=0)
		chain.fit(X_train, Y_train)
		store_data_as_pickle(chain, f'range-0-to-10000-{C}')
		y_pred = chain.predict(X_test)

		print(f'{C}\t{accuracy_score(Y_test, y_pred)}\t{f1_score(Y_test, y_pred, average="macro")}\t{precision_score(Y_test, y_pred, average="macro", zero_division=0)}\t{recall_score(Y_test, y_pred, average="macro")}')


	# chains = [ClassifierChain(base_clf, cv=folds, order='random', random_state=i) for i in range(10)]
	# for index, chain in enumerate(chains):
	# 	print(f'Fitting chain {index} ...')
	# 	chain.fit(X_train, Y_train)
	# 	print(f'Done fitting, storing as {CACHED_DATA_FILENAME}-{index}.pickle ...')
	# 	store_data_as_pickle(chains, f'{CACHED_DATA_FILENAME}-{index}')
	
