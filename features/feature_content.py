from util.msdialog_data_helper import parse_data
from util.pickle_helper import get_pickled_data, store_data_as_pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Path to content feature pickle
CACHED_DATA_FILENAME = 'content_pickle_data'

KEYWORDS = ["what", "where", "when", "why", "who", "how"]


# Calculates utterance similarity score between itself and the first utterance of the dialog.
def get_utterance_similarity(json_data):
    results = []
    for dialog_id, dialog_dict in json_data.items():
        dialog_data = dialog_dict['utterances']

        utterances = [item["utterance"] for item in dialog_data]
        tfidf = compute_cosine_similarity(utterances)
        # Get the cosine similarity score between each utterance and the first utterance of the dialog.
        for i in range(0, tfidf.shape[0]):
            results.append(tfidf[0, i])

    return results


# Calculates utterance similarity score between itself and the dialog.
def get_dialog_similarity(json_data):
    results = []
    for dialog_id, dialog_dict in json_data.items():
        dialog_data = dialog_dict['utterances']

        utterances = [item["utterance"] for item in dialog_data]
        tfidf = compute_cosine_similarity(utterances)
        # Compute the average cosine similarity score between each utterance and the entire dialog.
        for i in range(tfidf.shape[0]):
            sum = 0
            for j in range(tfidf.shape[0]):
                if i != j:
                    sum += tfidf[i, j]

            results.append(sum / (tfidf.shape[0] - 1))

    return results


# Computes the cosine similarity scores for a set of utterances.
def compute_cosine_similarity(utterances):
    tfidf = TfidfVectorizer().fit_transform(utterances)
    return tfidf * tfidf.T


# Checks whether a utterance contains a question mark.
def get_question_mark(json_data):
    all_utterances, _, _, _ = parse_data(json_data)
    results = []

    for utterance in all_utterances:
        if "?" in utterance:
            results.append(1)
        else:
            results.append(0)

    return results


# Get features that indicate duplication by checking 'same' and 'similar' words.
def get_duplicate(json_data):
    all_utterances, _, _, _ = parse_data(json_data)
    results = []

    for utterance in all_utterances:
        if "same" or "similar" in utterance.lower():
            results.append(1)
        else:
            results.append(0)

    return results


# Check for the 5W1H keywords in the utterances.
def get_keywords(json_data):
    all_utterances, _, _, _ = parse_data(json_data)
    results = []

    for utterance in all_utterances:
        keywords = [0] * 6

        for index, keyword in enumerate(KEYWORDS):
            if keyword in utterance.lower():
                keywords[index] = 1

        results.append(keywords)

    return results


# Calculate all features and return them
def calculate_content_features(json_data):
    utterance_similarity = get_utterance_similarity(json_data)
    dialog_similarity = get_dialog_similarity(json_data)
    question_mark = get_question_mark(json_data)
    duplicate = get_duplicate(json_data)
    keywords = get_keywords(json_data)

    return utterance_similarity, dialog_similarity, question_mark, duplicate, keywords


# Calculate all features and store it in a pickle file
def calculate_and_store_content_as_pickle(json_data, filename=CACHED_DATA_FILENAME):
    store_data_as_pickle(calculate_content_features(json_data), filename)


# Fetch the pickle file of the features
def fetch_content_features_pickle(filename=CACHED_DATA_FILENAME):
    return get_pickled_data(filename)
