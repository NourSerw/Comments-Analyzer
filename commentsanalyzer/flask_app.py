import configparser
import logging
import os

import nltk
from flask import request, jsonify
import praw
from flask import Flask, render_template, request, redirect, url_for, abort
from flask_bootstrap import Bootstrap
from joblib import load
import twitter_app
import json

# app = Flask(__name__, template_folder='templates')
app = Flask(__name__)
app.static_folder = 'static'
Bootstrap(app)
full_url = ''
submission = ''
# sm_key = {}
HEADER_PHOTO = os.path.join('static', 'img')
app.config['PHOTO'] = HEADER_PHOTO
sm_key = {"Platform": None,
          "Topic": None,
          "Source": None}


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
    if sm_key['Topic'] == 'General':
        file = open("reddit_classifier_FINAL_LOCAL.joblib", "rb")
        print("General Model")
    elif sm_key['Topic'] == 'News':
        file = open("reddit_classifier_FINAL_news_v0.joblib", "rb")
        print("News Model")
    elif sm_key['Topic'] == "Football":
        file = open("reddit_classifier_FINAL_football_v0.joblib", "rb")
        print("Football Model")
    clf = load(file)
    file.close()
    return clf


def get_vectorizer():
    if sm_key['Topic'] == 'General':
        file = open("TfidfVectorizer_vectorizer_LOCAL.joblib", "rb")
        print("General vectorizer")
    elif sm_key['Topic'] == 'News':
        file = open("TfidfVectorizer_vectorizer_news_v0.joblib", "rb")
        print("News vectorizer")
    elif sm_key['Topic'] == "Football":
        file = open("TfidfVectorizer_vectorizer_football_v0.joblib", "rb")
    vector = load(file)
    file.close()
    return vector


def get_prediction(posts, clf, vector, neg_weight, neu_weight, pos_weight):
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
    return get_percentage(neg_weight, neu_weight, pos_weight)


def get_percentage(neg_weight, neu_weight, pos_weight):
    total = neg_weight + neu_weight + pos_weight
    print("Percentage of sentiment as following: ")
    print("Negative: " + str(round((neg_weight / total) * 100, 2)))
    print("Neutral: " + str(round((neu_weight / total) * 100, 2)))
    print("Positive: " + str(round((pos_weight / total) * 100, 2)))
    values_dict = {
        "Topic": sm_key['Topic'],
        "neg_percentage": str(round((neg_weight / total) * 100, 2)),
        "neutral_percentage": str(round((neu_weight / total) * 100, 2)),
        "positive_percentage": str(round((pos_weight / total) * 100, 2)),
        "Post title": submission.title,
        "score": submission.score
    }

    return values_dict


def pipeline(url):
    print("URL: ", url)
    submission = reddit_credit(url)
    posts = get_comments(submission)
    return get_prediction(posts, get_clf(), get_vectorizer(), 0, 0, 0)


@app.route('/', methods=['POST', 'GET'])
def get_data():
    global sm_key
    global full_url
    if request.form.get("Dropdown") and request.form.get("Dropdown").strip():
        sm_key['Platform'] = 0
        sm_key['Topic'] = request.form.get("Dropdown")
    elif request.form.get("Dropdown_twitter") and request.form.get("Dropdown_twitter").strip():
        sm_key['Platform'] = 1
        sm_key['Topic'] = request.form.get("Dropdown_twitter")
        sm_key['Source'] = request.form.get("Dropdown_twitter_source")
    if request.method == 'POST':
        if sm_key['Platform'] == 0:
            print(sm_key)
            thread = request.form['Analyze']
            full_url = thread
            try:
                thread = thread.split('/', 8)[7]
                return redirect(url_for("success", name=thread))
            except:
                abort(404, "You have entered a wrong link, please go back and re-enter your link")
        elif sm_key['Platform'] == 1:
            thread = request.form['Analyze_twitter']
            if sm_key['Source'] == 'Hashtag':
                full_url = "hashtag_search?" + thread
                twitter_pipeline = {
                    "Topic": sm_key['Topic'],
                    "Source": sm_key['Source'],
                    "Data": thread
                }
            elif sm_key['Source'] == 'Singular_tweet':
                full_url = thread
                try:
                    twitter_pipeline = {
                            "Topic": sm_key['Topic'],
                            "Source": sm_key['Source'],
                            "Data": thread.split('/', 5)[5]
                        }
                    print("twitter_pipeline")
                    print(twitter_pipeline)
                    # return redirect(url_for("twitter_success", name=twitter_pipeline))
                except:
                    abort(404, "You have entered something wrong, please go back and re-enter your link/hashtag")
            return redirect(url_for("twitter_success", name=twitter_pipeline))

    else:
        return render_template('index.html')


@app.route('/reddit_success/v1/<name>')
def success(name):
    result = pipeline(full_url)
    return render_template('query_result.html', results=result)


@app.route('/twitter_success/v1/<name>')
def twitter_success(name):
    name = name.replace("'", "\"")
    name = json.loads(name)
    try:
        print(name)
        result = twitter_app.twitter_pipeline(name)
        return render_template('query_result.html', results=result)
    except:
        abort(404, "You have entered something wrong, please go back and re-enter your link/hashtag")


@app.route('/ca/reddit/v1/results', methods=['POST'])
def reddit_api():
    global sm_key
    content = request.json
    sm_key = {'Platform': 0, 'Topic': content['model'], 'Source': content['model']}
    result = pipeline(content['url'])
    return jsonify(result)


@app.route('/ca/twitter/v1/results', methods=['POST'])
def twitter_api():
    global sm_key
    content = request.json
    sm_key = {'Topic': content['model'], 'Source': content['source']}
    if content['source'] == 'Hashtag':
        sm_key['Data'] = content['url']
    elif content['source'] == 'Singular_tweet':
        sm_key['Data'] = content['url'].split('/', 5)[5]
    result = twitter_app.twitter_pipeline(sm_key)
    return jsonify(result)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/changelog')
def change_log():
    return render_template('changelog.html')


if __name__ == "__main__":
    app.run(debug=True)
