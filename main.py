import json

from features.feature_sentiment import calculate_sentimental_features
from util.msdialog_data_helper import parse_data, print_data_analytics
from features.feature_structural import calculate_structural_features

DATA_PATH = './data/MSDialog/MSDialog-Intent.json'

def load_data(data_path=DATA_PATH):
    with open(data_path, mode='r') as json_file:
        return json.load(json_file)

def feature_analysis(json_data):
    all_utterances, all_isUsers, all_tags, utterance_positions = parse_data(json_data)
    print_data_analytics(all_utterances, all_isUsers, all_tags)

    all_features = (all_utterances, all_isUsers, all_tags, utterance_positions)

    structural_features = calculate_structural_features(utterance_positions)
    sentiment_features = calculate_sentimental_features(all_features)

    print(sentiment_features)


if __name__ == "__main__":
    json_data = load_data()

    feature_analysis(json_data)
