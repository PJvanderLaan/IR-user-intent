import json
from scipy import sparse
import numpy as np
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
    utterance_positions, normalized_utterance_positions, utterance_lengths, unique_utterance_lengths, unique_stemmed_utterance_lengths, commented_by_starter \
        = fetch_structural_features_pickle()

    combined_features = combine_features([
        # content features
        # ...
        # structural features
        utterance_positions, 
        normalized_utterance_positions,
        utterance_lengths, 
        unique_utterance_lengths, 
        unique_stemmed_utterance_lengths, 
        commented_by_starter
        # sentiment features
        #...
    ])

if __name__ == "__main__":
    json_data = load_data()
    print_data_analytics(json_data)
    feature_analysis(json_data)
