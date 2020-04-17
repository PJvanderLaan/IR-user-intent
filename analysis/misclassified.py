from classifier.evaluation import get_accuracies
import numpy as np


def get_misclassified(predY, dataY):
    accs = get_accuracies(dataY, predY)
    np_accs = np.array(accs)

    return np_accs