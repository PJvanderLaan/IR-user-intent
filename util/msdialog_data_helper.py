from collections import Counter
import random

# Append all items of an array to an array
def appendAll(array, items):
	result = array
	for item in items:
		result.append(item)
	return result

# Check if the title equals user
def checkIfUser(title):
	return title.lower() == "user"

def fetch_Y_labels(json_data, mapping=['OQ', 'RQ', 'CQ', 'FD', 'FQ', 'IR', 'PA', 'PF', 'NF', 'GG', 'JK', 'O']):
	most_frequent = fetch_preprocessed_labels(json_data)
	
	labels = fetch_labels(json_data)
	result = []
	for label in labels:
		if label in most_frequent:
			result.append(label)
		else:
			indexes = [i for i, x in enumerate(label) if x == True]
			random_true_index = random.randint(0, len(indexes) - 1)
			random_label = list(map(lambda x: False, mapping))
			random_label[random_true_index] = True
			result.append(random_label)
	return result

# Fetch the labels as an array of one hot coding rows.
def fetch_labels(json_data, mapping=['OQ', 'RQ', 'CQ', 'FD', 'FQ', 'IR', 'PA', 'PF', 'NF', 'GG', 'JK', 'O']):
	blacklist_mapping = ['GG', 'JK', 'O']
	all_labels = []
	for dialog_id, dialog_dict in json_data.items(): 
		for item in dialog_dict['utterances']:
			labels = item["tags"].split(" ")
			labels = list(filter(lambda x: x in mapping, labels))
			one_hot_encoding = [x in labels for x in mapping]
			clean_one_hot_encoding = [x in labels and x not in blacklist_mapping for x in mapping]
			
			if (clean_one_hot_encoding.count(1) > 0):
				all_labels.append(clean_one_hot_encoding)
			else:
				all_labels.append(one_hot_encoding)
	return all_labels

# Parse the json data
def parse_data(json_data):
	all_utterances, all_isUsers, all_tags, all_utterance_positions = [], [], [], []
	for dialog_id, dialog_dict in json_data.items():
		utterances, isUser, tags, utterance_positions = parse_dialog(dialog_dict)

		all_utterances = appendAll(all_utterances, utterances)
		isUser = appendAll(all_isUsers, isUser)
		all_tags = appendAll(all_tags, tags)
		all_utterance_positions = appendAll(all_utterance_positions, utterance_positions)
	return all_utterances, all_isUsers, all_tags, all_utterance_positions

# Parse a dialog
def parse_dialog(dialog_dict):
	title = dialog_dict["title"]
	dialog_data = dialog_dict['utterances']

	# TODO: Utterance strings can be cleaned!
	utterances = [item["utterance"] for item in dialog_data]
	isUser = [checkIfUser(item["actor_type"]) for item in dialog_data]
	tags = [item["tags"].split(" ") for item in dialog_data]
	utterance_positions = [item["utterance_pos"] for item in dialog_data]

	return utterances, isUser, tags, utterance_positions

# Print data analytics
def print_data_analytics(json_data):
    all_utterances, all_isUsers, all_tags, _ = parse_data(json_data)
    blacklist_mapping = ['GG', 'JK', 'O']

    user_dialog_count = len(list(filter(lambda x: x == True, all_isUsers)))
    print(all_tags)
    all_tags = [", ".join(items) for items in all_tags]

    tags = list(Counter(all_tags).keys())
    counts = list(Counter(all_tags).values())
    print_number_top_tags = 32

    print(f'Total rows: {len(all_utterances)}')
    print(f'User: {user_dialog_count}, Agent: {len(all_utterances) - user_dialog_count}')
    print(f'Top {print_number_top_tags} tags with highest frequency:')    
    for i in range(0, print_number_top_tags):
        print(f' {i + 1}. {tags[i]} ({counts[i]})')

def fetch_preprocessed_labels(json_data, number_of_tags=32):
	others_ohe = [False, False, False, False, False, False, False, False, False, False, False, True]
	labels = fetch_labels(json_data)
	counted = Counter(tuple(item) for item in labels)
	most_frequent = counted.most_common(number_of_tags)
	most_frequent_list = list(map(lambda x: list(x[0]), most_frequent))
	most_frequent_list.append(others_ohe)
	return most_frequent_list

def print_most_frequent_labels(json_data):
	most_frequent = fetch_preprocessed_labels(json_data)
	
	rr = []
	for labels in most_frequent:
		indexes = [i for i, x in enumerate(labels) if x == True]
		things = list(map(lambda x: mapping[x], indexes))
		rr.append(things)
	
	for r in rr:
		print(r)



