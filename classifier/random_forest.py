from pandas import np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, cross_validate
import matplotlib.pyplot as plt
from classifier.evaluation import get_accuracy, get_precision, get_recall, get_f1


def rf_classifier(trainX, trainY, n_estimators, criterion):
    clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion)
    clf.fit(trainX, trainY)
    return clf


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
