import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, cross_validate
import matplotlib.pyplot as plt
from classifier.evaluation import get_accuracy, get_precision, get_recall, get_f1

scoring = {'Accuracy': make_scorer(get_accuracy),
           'Precision': make_scorer(get_precision),
           'Recall': make_scorer(get_recall),
           'F1-score': make_scorer(get_f1)}


def rf_classifier():
    # Parameter values are the optimal results from the experiments.
    clf = RandomForestClassifier(n_estimators=100, criterion='entropy', min_samples_split=4, min_samples_leaf=1,
                                 max_features=None, max_depth=10)
    return clf


def evaluate_random_forest(train, test):
    return cross_validate(rf_classifier(), train, test, cv=10, scoring=scoring)


def rf_predict(clf, dataX):
    return clf.predict(dataX)


# Hyperparam optimization using grid search plot for the criterion and n_estimators features.
def calculate_n_estimators(train, test):
    parameters = {
        'criterion': ['gini', 'entropy'],
        'n_estimators': [10, 20, 30, 50, 100, 200, 300],
    }
    scoring = make_scorer(get_accuracy)
    clf = GridSearchCV(RandomForestClassifier(), parameters, scoring=scoring, cv=10, verbose=10)
    clf.fit(train, test)

    scores_mean = clf.cv_results_['mean_test_score']
    params = clf.cv_results_['params']

    gini = []
    entropy = []
    x = np.arange(len(parameters['n_estimators']))
    width = 0.35

    for index, param in enumerate(params):
        if param['criterion'] == 'gini':
            gini.append(scores_mean[index])
        else:
            entropy.append(scores_mean[index])

    plt.bar(x - width / 2, gini, width, color='orange', align='center', label='gini')
    plt.bar(x + width / 2, entropy, width, align='center', label='entropy')
    plt.xticks(x, parameters['n_estimators'])
    plt.ylabel("Accuracy score")
    plt.xlabel("n_estimators")
    plt.legend()
    plt.title("Grid search of n_estimators and criterion")
    plt.show()

    print(clf.best_params_, clf.best_score_)


# Hyperparam optimization for min_samples_split
def calculate_min_samples_split(train, test):
    min_samples_split = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    accuracy = []
    precision = []
    recall = []
    f1 = []

    scoring = {'Accuracy': make_scorer(get_accuracy),
               'Precision': make_scorer(get_precision),
               'Recall': make_scorer(get_recall),
               'F1-score': make_scorer(get_f1)}

    for split in min_samples_split:
        print(split)
        clf = RandomForestClassifier(n_estimators=100, criterion='entropy', min_samples_split=split)
        values = cross_validate(clf, train, test, cv=10, scoring=scoring)
        accuracy.append(values['test_Accuracy'].mean())
        precision.append(values['test_Precision'].mean())
        recall.append(values['test_Recall'].mean())
        f1.append(values['test_F1-score'].mean())

    plt.plot(min_samples_split, accuracy, label='Accuracy')
    plt.plot(min_samples_split, precision, label='Precision')
    plt.plot(min_samples_split, recall, label='Recall')
    plt.plot(min_samples_split, f1, label='F1-score')
    plt.ylabel("Accuracy score")
    plt.xlabel("min_samples_split")
    plt.title("Minimum required samples for internal node split optimization")
    plt.legend()
    plt.show()

    print(max(accuracy), min_samples_split[accuracy.index(max(accuracy))])


# Hyperparam optimization for min_samples_leaf
def calculate_min_samples_leaf(train, test):
    min_samples_leaf = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    accuracy = []
    precision = []
    recall = []
    f1 = []

    for leaf in min_samples_leaf:
        print(leaf)
        clf = RandomForestClassifier(n_estimators=100, criterion='entropy', min_samples_split=4, min_samples_leaf=leaf)
        values = cross_validate(clf, train, test, cv=10, scoring=scoring)
        accuracy.append(values['test_Accuracy'].mean())
        precision.append(values['test_Precision'].mean())
        recall.append(values['test_Recall'].mean())
        f1.append(values['test_F1-score'].mean())

    plt.plot(min_samples_leaf, accuracy, label='Accuracy')
    plt.plot(min_samples_leaf, precision, label='Precision')
    plt.plot(min_samples_leaf, recall, label='Recall')
    plt.plot(min_samples_leaf, f1, label='F1-score')
    plt.ylabel("Accuracy score")
    plt.xlabel("min_samples_split")
    plt.title("Minimum required samples for internal node split optimization")
    plt.legend()
    plt.show()

    print(max(accuracy), min_samples_leaf[accuracy.index(max(accuracy))])


# Hyperparam optimization for max_features
def calculate_max_features(train, test):
    max_features = [None, 'auto', 'log2']
    x = range(3)
    accuracy = []
    precision = []
    recall = []
    f1 = []

    for feature in max_features:
        print(feature)
        clf = RandomForestClassifier(n_estimators=100, criterion='entropy', min_samples_split=4, min_samples_leaf=1,
                                     max_features=feature)
        values = cross_validate(clf, train, test, cv=10, scoring=scoring)
        accuracy.append(values['test_Accuracy'].mean())
        precision.append(values['test_Precision'].mean())
        recall.append(values['test_Recall'].mean())
        f1.append(values['test_F1-score'].mean())

    plt.bar(x, accuracy, 0.4, label='Accuracy')
    plt.bar(x, precision, 0.3, label='Precision')
    plt.bar(x, recall, 0.2, label='Recall')
    plt.bar(x, f1, 0.1, label='F1-score')
    plt.xticks(x, ['n_features', 'sqrt(n_features)', 'log2(n_features)'])
    plt.xlabel("max_features setting")
    plt.title("The number of features to consider for each split")
    plt.legend()
    plt.show()

    print(max(accuracy), max_features[accuracy.index(max(accuracy))])


# Hyperparam optimization for max_depth
def calculate_max_depth(train, test):
    max_depth = [5, 10, 20, 30, 40, 50]
    accuracy = []
    precision = []
    recall = []
    f1 = []

    for depth in max_depth:
        print(depth)
        clf = RandomForestClassifier(n_estimators=100, criterion='entropy', min_samples_split=4, min_samples_leaf=1,
                                     max_features=None, max_depth=depth)
        values = cross_validate(clf, train, test, cv=10, scoring=scoring)
        accuracy.append(values['test_Accuracy'].mean())
        precision.append(values['test_Precision'].mean())
        recall.append(values['test_Recall'].mean())
        f1.append(values['test_F1-score'].mean())

    plt.plot(max_depth, accuracy, label='Accuracy')
    plt.plot(max_depth, precision, label='Precision')
    plt.plot(max_depth, recall, label='Recall')
    plt.plot(max_depth, f1, label='F1-score')
    plt.xlabel("max_depth")
    plt.title("The maximum depth of the tree")
    plt.legend()
    plt.show()

    print(max(accuracy), max_depth[accuracy.index(max(accuracy))])
