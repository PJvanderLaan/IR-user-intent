import json
from scipy import sparse
import numpy as np

from features.feature_content import fetch_content_features_pickle, calculate_and_store_content_as_pickle
from features.feature_sentiment import calculate_sentimental_features, calculate_and_store_sentiment_as_pickle, \
    fetch_sentiment_features_pickle
from util.msdialog_data_helper import parse_data, print_data_analytics, fetch_labels
from features.feature_structural import calculate_and_store_as_pickle, fetch_structural_features_pickle
from classifier_chaining import chaining_svm

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
    chaining_svm(X_train, Y_train)
