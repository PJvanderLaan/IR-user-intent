import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt


def get_random_forrest_classifier(train, test):
    # Implementation is inspired from:
    # https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
    cls = ExtraTreesClassifier(n_estimators=100)
    cls.fit(train, test)
    importance_scores = cls.feature_importances_
    sd = np.std([tree.feature_importances_ for tree in cls.estimators_],
                axis=0)

    return importance_scores, sd


# def plot_feature_importance():
# plt.figure()
# plt.title("Feature importance scores")
# plt.bar(range(X.shape[1]), importances[indices],
#         color="r", yerr=std[indices], align="center")
# plt.xticks(range(X.shape[1]), indices)
# plt.xlim([-1, X.shape[1]])
# plt.show()

def analyze_feature_importance(train, test):
    importance_scores, sd = get_random_forrest_classifier(train, test)
    print(importance_scores)
    print(sd)
