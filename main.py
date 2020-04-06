import json
from scipy import sparse
import numpy as np
from sklearn.model_selection import KFold

from classifier.knn import knn_classifier, knn_predict, knn_get_accuracy, knn_get_recall, knn_get_precision, knn_get_f1
from features.feature_content import fetch_content_features_pickle, calculate_and_store_content_as_pickle
from features.feature_sentiment import calculate_sentimental_features, calculate_and_store_sentiment_as_pickle, \
    fetch_sentiment_features_pickle
from util.msdialog_data_helper import parse_data, print_data_analytics, fetch_labels
from features.feature_structural import calculate_and_store_as_pickle, fetch_structural_features_pickle

DATA_PATH = './data/MSDialog/MSDialog-Intent.json'


def load_data(data_path=DATA_PATH):
    with open(data_path, mode='r') as json_file:
        return json.load(json_file)

def construct_data(json_data):
    utterance_similarity, dialog_similarity, question_mark, duplicate, keywords = fetch_content_features_pickle()

    # Call calculate_and_store_as_pickle(json_data) to recalculate
    utterance_positions, normalized_utterance_positions, utterance_lengths, unique_utterance_lengths, unique_stemmed_utterance_lengths, commented_by_starter \
        = fetch_structural_features_pickle()

    # Call calculate_and_store_sentiment_as_pickle(parsed_data) to recalculate
    negative, neutral, positive, exclamation, thank, feedback, pos_score, neg_score = fetch_sentiment_features_pickle()

    # Combine the content, structural and sentiment features to a CSR matrix
    X_np_train = np.array([
        # content features
        utterance_similarity,
        dialog_similarity,
        question_mark,
        duplicate,
        # keywords,
        # structural features
        utterance_positions,
        normalized_utterance_positions,
        utterance_lengths,
        unique_utterance_lengths,
        unique_stemmed_utterance_lengths,
        commented_by_starter,
        # sentiment features
        negative,
        neutral,
        positive,
        exclamation,
        thank,
        feedback,
        pos_score,
        neg_score
    ]).T
    X_csr_train = sparse.csr_matrix(X_np_train)

    Y_np_train = np.array(fetch_labels(json_data))
    Y_csr_train = sparse.csr_matrix(Y_np_train)
    return X_csr_train, Y_csr_train, X_np_train, Y_np_train


if __name__ == "__main__":
    json_data = load_data()
    X_csr_train, Y_csr_train, X_np_train, Y_np_train = construct_data(json_data)
    kf = KFold(n_splits=5)
    kf.get_n_splits(X_np_train)
    for train_index, test_index in kf.split(X_np_train):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X_np_train[train_index], X_np_train[test_index]
        y_train, y_test = Y_np_train[train_index], Y_np_train[test_index]
        knn = knn_classifier(X_train, y_train, 3)
        knn_pred = knn_predict(knn, X_test)
        knn_prec = knn_get_precision(y_test, knn_pred)
        knn_recall = knn_get_recall(y_test, knn_pred)
        knn_f1 = knn_get_f1(y_test, knn_pred)
        knn_acc = knn_get_accuracy(y_test, knn_pred)
        print("Acc:{}\tRecall:{}\tPrecision:{}\tF1-score:{}".format(knn_acc, knn_recall, knn_prec, knn_f1))
    pass
