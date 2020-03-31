import json
from util.msdialog_data_helper import parse_data, print_data_analytics

DATA_PATH = './data/MSDialog/MSDialog-Intent.json'

def load_data(data_path=DATA_PATH):
    with open(data_path, mode='r') as json_file:
        return json.load(json_file)

if __name__ == "__main__":
    json_data = load_data()
    all_utterances, all_isUsers, all_tags = parse_data(json_data)
    print_data_analytics(all_utterances, all_isUsers, all_tags)