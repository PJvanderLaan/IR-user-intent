import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from util.pickle_helper import store_data_as_pickle
from classifier.evaluation import *

CACHED_DATA_FILENAME = 'fitted_chain'


def chaining_adaboost(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=0)

    base_clf = AdaBoostClassifier(algorithm="SAMME", n_estimators=200)
    chain = ClassifierChain(base_clf, cv=2, order='random', random_state=0)
    chain.fit(X_train, Y_train)
    y_pred = chain.predict(X_test)
    print(
        f'{get_accuracy(Y_test, y_pred)}\t{get_f1(Y_test, y_pred)}\t{get_recall(Y_test, y_pred)}\t{get_precision(Y_test, y_pred)}')


def chaining_svm(X, Y, max_iter=-1):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=0)

    Cs = np.logspace(-2, 10, 30)
    res = []
    print(f'Trying Cs: {Cs}')
    print('C \t accuracy \t f1 \t precision \t recall')
    for C in Cs:
        base_clf = SVC(C=C, kernel='rbf', max_iter=max_iter)

        chain = ClassifierChain(base_clf, cv=2, order='random', random_state=0)
        chain.fit(X_train, Y_train)
        y_pred = chain.predict(X_test)
        res.append([[get_accuracy(Y_test, y_pred), get_f1(Y_test, y_pred), get_recall(Y_test, y_pred),
                     get_precision(Y_test, y_pred)], C])
        print(
            f'{C}\t{get_accuracy(Y_test, y_pred)}\t{get_f1(Y_test, y_pred)}\t{get_recall(Y_test, y_pred)}\t{get_precision(Y_test, y_pred)}')

    store_data_as_pickle(res, f'svm-chain-logscale-values')

    acc = np.asarray([[a[0][0], a[1]] for a in res])
    f1 = np.asarray([[a[0][1], a[1]] for a in res])
    recall = np.asarray([[a[0][2], a[1]] for a in res])
    precision = np.asarray([[a[0][3], a[1]] for a in res])

    print("Max acc without question at default_dist: ", acc[np.argmax(acc[:, 0]), 1], " ", np.max(acc[:, 0]))
    print("Max f1 without question at default_dist: ", f1[np.argmax(f1[:, 0]), 1], " ", np.max(f1[:, 0]))
    print("Max recall without question at default_dist: ", recall[np.argmax(recall[:, 0]), 1], " ",
          np.max(recall[:, 0]))
    print("Max precision without question at default_dist: ", precision[np.argmax(precision[:, 0]), 1], " ",
          np.max(precision[:, 0]))
    plt.plot(acc[:, 1], acc[:, 0], label='Accuracy')
    plt.plot(f1[:, 1], f1[:, 0], label='F1-Score')
    plt.plot(recall[:, 1], recall[:, 0], label='Recall')
    plt.plot(precision[:, 1], precision[:, 0], label='Precision')
    plt.legend()
    plt.xscale('log')
    plt.xlabel("C regularization parameter")
    plt.title("SVM with ClassifierChain 10 folds")
    plt.show()
