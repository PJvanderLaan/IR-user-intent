import json

DATA_PATH = './data/MSDialog/MSDialog-Intent.json'

def load_data(data_path=DATA_PATH):
    with open(data_path, mode='r') as json_file:
        return json.load(json_file)

if __name__ == "__main__":
    print(load_data())