from collections import Counter

def appendAll(array, items):
	result = array
	for item in items:
		result.append(item)
	return result

def checkIfUser(title):
	return title.lower() == "user"

def parse_data(json_data):
	all_utterances, all_isUsers, all_tags = [], [], []
	for dialog_id, dialog_dict in json_data.items():
		utterances, isUser, tags = parse_dialog(dialog_dict)

		all_utterances = appendAll(all_utterances, utterances)
		isUser = appendAll(all_isUsers, isUser)
		all_tags = appendAll(all_tags, tags)
	return all_utterances, all_isUsers, all_tags
				
def parse_dialog(dialog_dict):
	title = dialog_dict["title"]
	dialog_data = dialog_dict['utterances']

	# TODO: Utterance strings can be cleaned!
	utterances = [item["utterance"] for item in dialog_data]
	isUser = [checkIfUser(item["actor_type"]) for item in dialog_data]
	tags = [item["tags"].split(" ") for item in dialog_data]

	return utterances, isUser, tags

def print_data_analytics(all_utterances, all_isUsers, all_tags):
    user_dialog_count = len(list(filter(lambda x: x == True, all_isUsers)))
    all_tags = ["_".join(items) for items in all_tags]
    tags = list(Counter(all_tags).keys())
    counts = list(Counter(all_tags).values())
    print_number_top_tags = 10

    print(f'Total rows: {len(all_utterances)}')
    print(f'User: {user_dialog_count}, Agent: {len(all_utterances) - user_dialog_count}')
    for i in range(0, print_number_top_tags):
        print(f' > {tags[i]}: {counts[i]}')
    print(f' {print_number_top_tags} tags with highest frequency printed')