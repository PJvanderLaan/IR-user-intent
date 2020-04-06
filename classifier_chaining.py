import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.multioutput import ClassifierChain
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import jaccard_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# def chaining_svm(X_train, Y_train, C=0.5, gamma='scale', max_iter=-1, folds=10):
# 	print(X_train.shape)
# 	print(Y_train.shape)

# 	base_clf = SVC(C=C, kernel='rbf', gamma=gamma, max_iter=max_iter)

# 	chains = [ClassifierChain(base_clf, cv=folds, order='random', random_state=i) for i in range(10)]
# 	for chain in chains:
# 		chain.fit(X_train, Y_train)

def chaining_svm(X_train, Y_train, C=0.5, gamma='scale', max_iter=-1, folds=10):
	X, Y = fetch_openml('yeast', version=4, return_X_y=True)
	Y = Y == 'TRUE'
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2,
	                                                    random_state=0)

	print(X)
	print(Y)
	print(X.shape)
	print(Y.shape)

	# # Fit an independent logistic regression model for each class using the
	# # OneVsRestClassifier wrapper.
	# base_lr = LogisticRegression()
	# ovr = OneVsRestClassifier(base_lr)
	# ovr.fit(X_train, Y_train)
	# Y_pred_ovr = ovr.predict(X_test)
	# ovr_jaccard_score = jaccard_score(Y_test, Y_pred_ovr, average='samples')

	# # Fit an ensemble of logistic regression classifier chains and take the
	# # take the average prediction of all the chains.
	# chains = [ClassifierChain(base_lr, order='random', random_state=i) for i in range(10)]
	# for chain in chains:
	# 	chain.fit(X_train, Y_train)

	# Y_pred_chains = np.array([chain.predict(X_test) for chain in chains])
	# chain_jaccard_scores = [jaccard_score(Y_test, Y_pred_chain >= .5, average='samples') for Y_pred_chain in Y_pred_chains]

	# Y_pred_ensemble = Y_pred_chains.mean(axis=0)
	# ensemble_jaccard_score = jaccard_score(Y_test, Y_pred_ensemble >= .5, average='samples')

	# model_scores = [ovr_jaccard_score] + chain_jaccard_scores
	# model_scores.append(ensemble_jaccard_score)

	# model_names = ('Independent',
	#                'Chain 1',
	#                'Chain 2',
	#                'Chain 3',
	#                'Chain 4',
	#                'Chain 5',
	#                'Chain 6',
	#                'Chain 7',
	#                'Chain 8',
	#                'Chain 9',
	#                'Chain 10',
	#                'Ensemble')

	# x_pos = np.arange(len(model_names))

	# # Plot the Jaccard similarity scores for the independent model, each of the
	# # chains, and the ensemble (note that the vertical axis on this plot does
	# # not begin at 0).

	# fig, ax = plt.subplots(figsize=(7, 4))
	# ax.grid(True)
	# ax.set_title('Classifier Chain Ensemble Performance Comparison')
	# ax.set_xticks(x_pos)
	# ax.set_xticklabels(model_names, rotation='vertical')
	# ax.set_ylabel('Jaccard Similarity Score')
	# ax.set_ylim([min(model_scores) * .9, max(model_scores) * 1.1])
	# colors = ['r'] + ['b'] * len(chain_jaccard_scores) + ['g']
	# ax.bar(x_pos, model_scores, alpha=0.5, color=colors)
	# plt.tight_layout()
	# plt.show()

if __name__ == "__main__":
	print("classifier chaining")