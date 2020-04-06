import json
from scipy import sparse
import numpy as np

from features.feature_content import fetch_content_features_pickle, calculate_and_store_content_as_pickle
from features.feature_sentiment import calculate_sentimental_features, calculate_and_store_sentiment_as_pickle, \
    fetch_sentiment_features_pickle
from util.msdialog_data_helper import parse_data, print_data_analytics
from features.feature_structural import calculate_and_store_as_pickle, fetch_structural_features_pickle

DATA_PATH = './data/MSDialog/MSDialog-Intent.json'


def load_data(data_path=DATA_PATH):
    with open(data_path, mode='r') as json_file:
        return json.load(json_file)


def feature_to_csr(feature):
    return sparse.csr_matrix(np.array([feature]).T)


def combine_features(array_features):
    csr_features = list(map(lambda x: feature_to_csr(x), array_features))
    return sparse.hstack((csr_features))


def feature_analysis(json_data):
    utterance_similarity, dialog_similarity, question_mark, duplicate, keywords = fetch_content_features_pickle()

    # Fetch the structural features.
    # Call calculate_and_store_as_pickle(json_data) to recalculate
    utterance_positions, normalized_utterance_positions, utterance_lengths, unique_utterance_lengths, unique_stemmed_utterance_lengths, commented_by_starter \
        = fetch_structural_features_pickle()

    parsed_data = parse_data(json_data)
    # Call calculate_and_store_sentiment_as_pickle(parsed_data) to recalculate
    # calculate_and_store_sentiment_as_pickle(parsed_data)
    negative, neutral, positive, exclamation, thank, feedback, pos_score, neg_score = fetch_sentiment_features_pickle()

    # Combine the content, structural and sentiment features to a CSR matrix
    combined_features = combine_features([
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
    ])


if __name__ == "__main__":
    json_data = load_data()
    feature_analysis(json_data)
