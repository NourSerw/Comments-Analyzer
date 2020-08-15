from joblib import load


def main(submission, url):
    posts = get_comments(submission)
    return get_prediction(posts, get_clf(), get_vector(), 0, 0, 0)


def get_clf():
    file = open("reddit_classifier_FINAL_news.joblib")
    clf = load(file)
    file.close()
    return clf


def get_vector():
    file = open("TfidfVectorizer_vectorizer_news.joblib")
    vector = load(file)
    file.close()
    return vector


def get_comments(submission):
    posts = []
    submission.comments.replace_more(limit=None)
    for comment in submission.comments.list():
        posts.append(comment.body)
    if len(posts) > 1:
        return posts


def get_prediction(posts, clf, vector, neg_weight, neu_weight, pos_weight):
    for comment in posts:
        review_vector = vector.transform([comment])
        label = clf.predict(review_vector)
        if label == -1:
            neg_weight += 1
        elif label == 0:
            neu_weight += 1
        elif label == 1:
            pos_weight += 1
        print("Negative = " + str(neg_weight))
        print("Neutral = " + str(neu_weight))
        print("positive = " + str(pos_weight))
        print("------------------------------")
    return get_percentage(neg_weight, neu_weight, pos_weight)


def get_percentage(neg_weight, neu_weight, pos_weight):
    total = neg_weight + neu_weight + pos_weight
    print("Percentage of sentiment as following: ")
    print("Negative: " + str(round((neg_weight / total) * 100, 2)))
    print("Neutral: " + str(round((neu_weight / total) * 100, 2)))
    print("Positive: " + str(round((pos_weight / total) * 100, 2)))
    values_dict = {
        "Negative": neg_weight,
        "Neutral": neu_weight,
        "Positive": pos_weight
    }
    return values_dict
