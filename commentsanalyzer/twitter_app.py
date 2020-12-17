from joblib import load
import os
import nltk
import requests
import configparser

data_dict_local = {}


def get_clf():
    if data_dict_local['Topic'] == 'General':
        file = open("twitter_classifier_FINAL_general_v0.joblib", "rb")
        print("General Model_Twitter")
        clf = load(file)
    return clf


def get_vectorizer():
    if data_dict_local['Topic'] == 'General':
        file = open("twitter_vectorizer_general_v0.joblib", "rb")
        print("General vectorizer_twitter")
        vector = load(file)
    return vector


def twitter_pipeline(data_dict):
    global data_dict_local
    data_dict_local = data_dict
    if data_dict['Source'] == "Hashtag":
        posts = get_hashtag(data_dict["Data"])
        return twitter_prediction(posts)


def get_hashtag(data):
    url = "https://api.twitter.com/2/tweets/search/recent?query="
    twitter_params = data + " -is:retweet"
    url = url + twitter_params
    config = configparser.ConfigParser(interpolation=None)
    config.read("F:\\dev\\.git\\twitter_config.ini")
    headers = {"Authorization": "Bearer {}".format(str(config.get('TwitterCredit', 'bearer_token')))}
    response = requests.request("GET", url, headers=headers)
    res_json = response.json()
    posts = []
    for i in range(0, len(res_json['data'])):
        posts.append(res_json['data'][i]['text'])
    return posts


def twitter_prediction(posts):
    stopwords = nltk.corpus.stopwords.words('english')
    clf = get_clf()
    vector = get_vectorizer()
    neg_weight, neu_weight, pos_weight = 0, 0, 0
    for tweet in posts:
        review_vector = vector.transform([tweet])
        label = clf.predict(review_vector)
        if label == -1:
            neg_weight += 1
        elif label == 0:
            neu_weight += 1
        elif label == 1:
            pos_weight += 1
        words = nltk.word_tokenize(tweet)
        words = [word for word in words if len(word) > 1]
        words = [word for word in words if not word.isnumeric()]
        words = [word.lower() for word in words]
        words = [word for word in words if word not in stopwords]
        fdist = nltk.FreqDist(words)
    most_common = fdist.most_common(10)
    return get_percentage(neg_weight, neu_weight, pos_weight, most_common)


def get_percentage(neg_weight, neu_weight, pos_weight, most_common):
    total = neg_weight + neu_weight + pos_weight
    print("Percentage of sentiment as following: ")
    print("Negative: " + str(round((neg_weight / total) * 100, 2)))
    print("Neutral: " + str(round((neu_weight / total) * 100, 2)))
    print("Positive: " + str(round((pos_weight / total) * 100, 2)))
    values_dict = {
        "Total": total,
        "Negative": neg_weight,
        "Neutral": neu_weight,
        "Positive": pos_weight,
        "Topic": data_dict_local['Topic'],
        "neg_percentage": str(round((neg_weight / total) * 100, 2)),
        "neutral_percentage": str(round((neu_weight / total) * 100, 2)),
        "positive_percentage": str(round((pos_weight / total) * 100, 2)),
        "most_common_words": most_common
    }

    if data_dict_local['Source'] == "Hashtag":
        values_dict['Hashtag'] = data_dict_local['Data']
    elif data_dict_local['Source'] == "Tweet":
        values_dict['Tweet'] = data_dict_local['Data']


# print("Type of values_dict: ", type(values_dict))

    return values_dict
