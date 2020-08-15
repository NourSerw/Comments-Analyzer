from flask import Flask, render_template, request, redirect, url_for
from joblib import load
# from pickle import load
import praw
import os
import news_prediction
import configparser
import logging

app = Flask(__name__, template_folder='templates')

full_url = ''
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


@app.route("/")
def main():
    photo = os.path.join(app.config['PHOTO'], 'pp.jpg')
    return render_template('home_page.html', header_image=photo)


def reddit_credit(url):
    Config = configparser.ConfigParser()
    Config.read("F:\\dev\\.git\\config_flask.ini")
    try:
        reddit = praw.Reddit(user_agent=str(Config.get('RedditCredit', 'user_agent')),
                             client_id=str(Config.get('RedditCredit', 'client_id')),
                             client_secret=str(Config.get('RedditCredit', 'client_secret')))
    except Exception as e:
        logging.exception("Excpetion occured - reddit_credit()")
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
    file = open("reddit_classifier_FINAL_LOCAL.joblib", "rb")
    clf = load(file)
    file.close()
    return clf


def get_vectorizer():
    file = open("TfidfVectorizer_vectorizer_LOCAL.joblib", "rb")
    vector = load(file)
    file.close()
    return vector


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
    logging.info("values_dict")
    print(values_dict)
    return values_dict


def pipeline(url):
    submission = reddit_credit(url)
    posts = get_comments(submission)
    return get_prediction(posts, get_clf(), get_vectorizer(), 0, 0, 0)


@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        thread = request.form['Analyze']
        global full_url
        full_url = thread
        thread = thread.split('/', 8)[7]
        return redirect(url_for('success', name=thread))


@app.route('/success/<name>')
def success(name):
    return "<xmp>" + str(pipeline(full_url)) + " </xmp> "


if __name__ == "__main__":
    app.run()
