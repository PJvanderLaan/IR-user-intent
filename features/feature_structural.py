from util.msdialog_data_helper import parse_data, print_data_analytics, appendAll
from util.pickle_helper import get_pickled_data, store_data_as_pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

CACHED_DATA_FILENAME = 'structural_pickle_data'

stemmer = nltk.PorterStemmer()
nltk_stopwords = stopwords.words()

# Given a dialog, check for each utterance if it is commented by the initial commenter
def calculate_commented_by_starter(dialog_data):
	starter = dialog_data[0]['user_id']
	return [item['user_id'] == starter for item in dialog_data]

# Fetch the normalized utterance positions and if the utterance is commented by the starter 
def fetch_custom_feature_data(json_data):
	normalized_utterance_positions = []
	started_by_starters = []
	for dialog_id, dialog_dict in json_data.items():
		dialog_data = dialog_dict['utterances']
		normalized = [item['utterance_pos']/len(dialog_data) for item in dialog_data]
		normalized_utterance_positions = appendAll(normalized_utterance_positions, normalized)
		
		started_by_starter = calculate_commented_by_starter(dialog_data)
		started_by_starters = appendAll(started_by_starters, started_by_starter)

	return normalized_utterance_positions, started_by_starters

# Fetch several lengths of the utterances
def fetch_utterance_lengths(utterances):
	utterances_default = []
	utterances_unique = []
	utterances_unique_stemmed = []

	# without stopwords!
	for index, utterance in enumerate(utterances):
		if index % 100 == 0:
			print(f'Utterance lengths calculation progress: {index}/{len(utterances)}')
		text_tokens = word_tokenize(utterance)
		tokens_default = [word for word in text_tokens if not word in nltk_stopwords]
		utterances_default.append(len(tokens_default))

		tokens_unique = list(set(tokens_default))
		utterances_unique.append(len(tokens_unique))

		tokens_unique_stemmed = [stemmer.stem(token) for token in tokens_unique]
		utterances_unique_stemmed.append(len(tokens_unique_stemmed))

	return utterances_default, utterances_unique, utterances_unique_stemmed

# Calculate all features and return them
def calculate_structural_features(json_data):
	print("[!!] Run download_stopwords.py ONCE to download the required NLTK stopwords!")

	all_utterances, _, _, utterance_positions = parse_data(json_data)
	normalized_utterance_positions, commented_by_starter = fetch_custom_feature_data(json_data)
	utterance_lengths, unique_utterance_lengths, unique_stemmed_utterance_lengths = fetch_utterance_lengths(all_utterances)
	return utterance_positions, normalized_utterance_positions, utterance_lengths, unique_utterance_lengths, unique_stemmed_utterance_lengths, commented_by_starter

# Calculate all features and store it in a pickle file
def calculate_and_store_as_pickle(json_data, filename=CACHED_DATA_FILENAME):
	store_data_as_pickle(calculate_structural_features(json_data), filename)

# Fetch the pickle file of the features
def fetch_structural_features_pickle(filename=CACHED_DATA_FILENAME):
	return get_pickled_data(filename)

