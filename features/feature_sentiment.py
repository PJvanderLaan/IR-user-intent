from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# This method creates the sentiment features on the utterances. The order of the arrays is:
# negative, neutral, positive, exclamation, thank, feedback, lexicon
def calculate_sentimental_features(json_data, exclamation_binary=True, opinion_normalized=False):
    utterances = json_data[0]
    negative, neutral, positive = get_vader_features(utterances)
    exclamation = get_exclamationmark_features(utterances, exclamation_binary)
    thank = get_thank_feature(utterances)
    feedback = get_feedback(utterances)
    poa_score, neg_score = get_opinion_lexicon(utterances, ["good"], ["bad"], opinion_normalized)
    return negative, neutral, positive, exclamation, thank, feedback, poa_score, neg_score


# Get the VADER features from the utterances, returns negative, neutral, and positive feature arrays in that order
def get_vader_features(text):
    analyser = SentimentIntensityAnalyzer()
    negative = []
    neutral = []
    positive = []
    for utterance in text:
        scores = analyser.polarity_scores(utterance)
        negative.append(scores['neg'])
        neutral.append(scores['neu'])
        positive.append(scores['pos'])

    return negative, neutral, positive


# Get the exclamation mark feature. Has an option to do a binary or actual count of exclamation features so it will
# return 0 or 1 for binary=True and 0,1,2,... for binary=False. Default is binary=True
def get_exclamationmark_features(text, binary=True):
    exclamation = []
    for utterance in text:
        if binary:
            exclamation.append(1 if (utterance.find("!") != -1) else 0)
        else:
            exclamation.append(utterance.count("!"))
    return exclamation


# Get the thank feature which says if the word 'thank' is in the utterance
def get_thank_feature(text):
    thank = []
    for utterance in text:
        thank.append(1 if utterance.find("thank") != -1 else 0)
    return thank


# Get the feedback feature which says if the utterance contains either 'does not' or 'did not'
def get_feedback(text):
    feedback = []
    for utterance in text:
        feedback.append(1 if (utterance.find("does not") != -1) or (utterance.find("did not") != -1) else 0)
    return feedback


# Get the opinion lexicon feature, which gives the amount of positive and negative words in the utterance.
# If normalized=True then the count will be divided by the sentence length in words.
def get_opinion_lexicon(text, pos_lexicon, neg_lexicon, normalized=False):
    positive_count = []
    negative_count = []
    # For each utterance
    num_utterances = len(text)
    for i in range(0, num_utterances):
        # Tokenize utterance to list of words
        split = text[i].split()
        # Set initial count to 0
        positive_count.append(0)
        negative_count.append(0)
        # For each word in the positive lexicon
        for pos in pos_lexicon:
            positive_count[i] += split.count(pos)
        # For each word in the negative lexicon
        for neg in neg_lexicon:
            negative_count[i] += split.count(neg)
        if normalized:
            positive_count[i] /= len(split)
            negative_count[i] /= len(split)

    return positive_count, negative_count
