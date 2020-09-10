from flask import Flask, render_template, request, redirect, url_for, jsonify
from joblib import load
# from pickle import load
import praw
import os
import configparser
import logging
import nltk
from nltk.probability import FreqDist
from flask_bootstrap import Bootstrap

# app = Flask(__name__, template_folder='templates')
app = Flask(__name__)
Bootstrap(app)
full_url = ''
topic = ''
submission = ''
HEADER_PHOTO = os.path.join('static', 'img')
app.config['PHOTO'] = HEADER_PHOTO
logging.debug('This is a debug message')
logging.info('This is an info message')
logging.warning('This is a warning message')
logging.error('This is an error message')
logging.critical('This is a critical message')
logging.basicConfig(filename='commentsanalyzer_flask.log', filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger("commentsanalyzer_flask")


# @app.route("/")
# def main():
#    return render_template('index.html')


def reddit_credit(url):
    Config = configparser.ConfigParser()
    Config.read("F:\\dev\\.git\\config_flask.ini")
    try:
        reddit = praw.Reddit(user_agent=str(Config.get('RedditCredit', 'user_agent')),
                             client_id=str(Config.get('RedditCredit', 'client_id')),
                             client_secret=str(Config.get('RedditCredit', 'client_secret')))
    except Exception as e:
        logging.exception("Excpetion occured - reddit_credit()")
    global submission
    submission = reddit.submission(url=url)
    if submission is None:
        logging.critical('Reddit instance was not retrieved, please check credentials')
    else:
        logging.info("Reddit instance retrieved")
        return submission


def get_comments(submission):
    posts = []
    submission.comments.replace_more(limit=None)
    for comment in submission.comments.list():
        posts.append(comment.body)
    if len(posts) > 1:
        return posts


def get_clf():
    if topic == 'General':
        file = open("reddit_classifier_FINAL_LOCAL.joblib", "rb")
        print("General Model")
    elif topic == 'News':
        file = open("reddit_classifier_FINAL_news_v0.joblib", "rb")
        print("News Model")
    elif topic == "Football":
        file = open("reddit_classifier_FINAL_football_v0.joblib", "rb")
        print("Football Model")
    clf = load(file)
    file.close()
    return clf


def get_vectorizer():
    if topic == 'General':
        file = open("TfidfVectorizer_vectorizer_LOCAL.joblib", "rb")
        print("General vectorizer")
    elif topic == 'News':
        file = open("TfidfVectorizer_vectorizer_news_v0.joblib", "rb")
        print("News vectorizer")
    elif topic == "Football":
        file = open("TfidfVectorizer_vectorizer_football_v0.joblib", "rb")
    vector = load(file)
    file.close()
    return vector


def get_prediction(posts, clf, vector, neg_weight, neu_weight, pos_weight):
    fdsit = FreqDist()
    stopwords = nltk.corpus.stopwords.words('english')
    for comment in posts:
        review_vector = vector.transform([comment])
        label = clf.predict(review_vector)
        if label == -1:
            neg_weight += 1
        elif label == 0:
            neu_weight += 1
        elif label == 1:
            pos_weight += 1
        allWords = nltk.tokenize.word_tokenize(comment)
        allWordExceptStopDist = nltk.FreqDist(w.lower() for w in allWords if w not in stopwords)
        allWordExceptStopDist_02 = nltk.FreqDist(w.lower() for w in allWordExceptStopDist if w.isalnum())
        mostCommon = allWordExceptStopDist_02.most_common(10)
    return get_percentage(neg_weight, neu_weight, pos_weight, mostCommon)


def get_percentage(neg_weight, neu_weight, pos_weight, mostCommon):
    total = neg_weight + neu_weight + pos_weight
    print("Percentage of sentiment as following: ")
    print("Negative: " + str(round((neg_weight / total) * 100, 2)))
    print("Neutral: " + str(round((neu_weight / total) * 100, 2)))
    print("Positive: " + str(round((pos_weight / total) * 100, 2)))
    values_dict = [{
        "Total": total,
        "Negative": neg_weight,
        "Neutral": neu_weight,
        "Positive": pos_weight,
        "Topic": topic
    },
        {
            "neg_percentage": str(round((neg_weight / total) * 100, 2)),
            "neutral_percentage": str(round((neu_weight / total) * 100, 2)),
            "positive_percentage": str(round((pos_weight / total) * 100, 2)),
            "most_common_words": mostCommon
        },
        {
            "Post title": submission.title,
            "score": submission.score
        }
    ]
    logging.info("values_dict")
    return values_dict


def pipeline(url):
    submission = reddit_credit(url)
    posts = get_comments(submission)
    return get_prediction(posts, get_clf(), get_vectorizer(), 0, 0, 0)


@app.route('/', methods=['POST', 'GET'])
def get_data():
    global topic
    topic = request.form.get("Dropdown")
    if request.method == 'POST':
        print(topic)
        thread = request.form['Analyze']
        global full_url
        full_url = thread
        thread = thread.split('/', 8)[7]
        return redirect(url_for("success", name=thread))
    else:
        return render_template('index.html')


@app.route('/success/v1/<name>')
def success(name):
    return jsonify(pipeline(full_url))
    #render_template("query_result.html")


if __name__ == "__main__":
    app.run(debug=True)
