from collections import Counter

# Append all items of an array to an array
def appendAll(array, items):
	result = array
	for item in items:
		result.append(item)
	return result

# Check if the title equals user
def checkIfUser(title):
	return title.lower() == "user"

# Fetch the labels as an array of one hot coding rows.
def fetch_labels(json_data, mapping=['OQ', 'RQ', 'CQ', 'FD', 'FQ', 'IR', 'PA', 'PF', 'NF', 'GG', 'JK', 'O']):
	blacklist_mapping = ['GG', 'JK', 'O']
	all_labels = []
	for dialog_id, dialog_dict in json_data.items(): 
		for item in dialog_dict['utterances']:
			labels = item["tags"].split(" ")
			labels = list(filter(lambda x: x in mapping, labels))
			one_hot_encoding = [x in labels for x in mapping]
			clean_one_hot_encoding  = [x in labels and x not in blacklist_mapping for x in mapping]
			
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

    user_dialog_count = len(list(filter(lambda x: x == True, all_isUsers)))
    all_tags = [", ".join(items) for items in all_tags]
    tags = list(Counter(all_tags).keys())
    counts = list(Counter(all_tags).values())
    print_number_top_tags = 10

    print(f'Total rows: {len(all_utterances)}')
    print(f'User: {user_dialog_count}, Agent: {len(all_utterances) - user_dialog_count}')
    print(f'Top {print_number_top_tags} tags with highest frequency:')    
    for i in range(0, print_number_top_tags):
        print(f' {i + 1}. {tags[i]} ({counts[i]})')
