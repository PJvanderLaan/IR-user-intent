import json
from scipy import sparse
import numpy as np

from analysis.feature_importance import analyze_feature_importance
from features.feature_content import fetch_content_features_pickle, calculate_and_store_content_as_pickle
from features.feature_sentiment import calculate_sentimental_features, calculate_and_store_sentiment_as_pickle, \
    fetch_sentiment_features_pickle
from util.msdialog_data_helper import parse_data, print_data_analytics, fetch_labels
from features.feature_structural import calculate_and_store_as_pickle, fetch_structural_features_pickle

DATA_PATH = './data/MSDialog/MSDialog-Intent.json'

FEATURE_NAMES = [
    "Initial Utterance Similarity",
    "Dialog Similarity",
    "Question Mark",
    "Duplicate",
    "5W1H: What",
    "5W1H: Where",
    "5W1H: When",
    "5W1H: Why",
    "5W1H: Who",
    "5W1H: How",
    "Absolute Position",
    "Normalized Position",
    "Utterance Length",
    "Utterance Length Unique",
    "Utterance Length Stemmed Unique",
    "Is Starter",
    "Thank",
    "Exclamation Mark",
    "Feedback",
    "Sentiment Scores:",
    "Sentiment Scores:",
    "Sentiment Scores:",
    "Opinion Lexicon:",
    "Opinion Lexicon:"
]


def load_data(data_path=DATA_PATH):
    with open(data_path, mode='r') as json_file:
        return json.load(json_file)


def construct_data(json_data):
    utterance_similarity, dialog_similarity, question_mark, duplicate, keyword_what, keyword_where, \
    keywords_when, keywords_why, keywords_who, keywords_how = fetch_content_features_pickle()

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
        keyword_what,
        keyword_where,
        keywords_when,
        keywords_why,
        keywords_who,
        keywords_how,
        # structural features
        utterance_positions,
        normalized_utterance_positions,
        utterance_lengths,
        unique_utterance_lengths,
        unique_stemmed_utterance_lengths,
        commented_by_starter,
        # sentiment features
        thank,
        exclamation,
        feedback,
        negative,
        neutral,
        positive,
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

    # Feature importance analysis.
    analyze_feature_importance(X_np_train, Y_np_train, FEATURE_NAMES)
