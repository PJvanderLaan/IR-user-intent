import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from classifier.random_forest import rf_classifier

# Fit random forest classifier and calculate importance scores.
def get_random_forest_classifier_results(train, test):
    # This implementation is based on:
    # https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
    cls = rf_classifier(train, test, n_estimators=200, criterion='gini')

    # Calculate importance scores and standard deviation values for each tree.
    importance_scores = cls.feature_importances_
    sd = np.std([tree.feature_importances_ for tree in cls.estimators_],
                axis=0)

    return importance_scores, sd


# Get and print a sorted list with importance scores, standard deviations and feature names.
def get_feature_importance_results(importance_scores, sd, feature_names):
    results = []

    for index, score in enumerate(importance_scores):
        results.append([score, sd[index], feature_names[index]])

    results.sort(key=lambda x: x[0], reverse=True)

    pprint(results)
    return results


# Plot horizontal bar chart with the importance score for each feature.
def plot_feature_importance(feature_results):
    plt.figure()
    plt.title("Feature importance scores")
    x = range(len(feature_results))
    plt.barh(x, [x[0] for x in feature_results], xerr=[x[1] for x in feature_results])
    plt.xlabel("Importance score")
    plt.ylabel("Feature")
    plt.yticks(x, [x[2] for x in feature_results])
    plt.show()


# Analyze feature importance.
def analyze_feature_importance(train, test, feature_names):
    importance_scores, sd = get_random_forest_classifier_results(train, test)
    results = get_feature_importance_results(importance_scores, sd, feature_names)
    plot_feature_importance(results)
