import re
import string

import nltk
import pandas as pd
from joblib import dump
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from nltk.tokenize import TweetTokenizer


def main():
    df = pd.read_csv("F://Twitter Data//twitter_football_dataframe.csv")
    print("CSV loaded")
    print(df.shape)
    df.dropna(inplace=True)
    print(df.isna().sum())
    tk = TweetTokenizer(reduce_len=True)
    df.Tweet = df.apply(lambda row: tk.tokenize(row['Tweet']), axis=1)
    stop = set(stopwords.words('english'))
    punctuation = list(string.punctuation)
    stop.update(punctuation)
    global stop_words
    stop_words = stopwords.words('english')
    global lemmatizer
    lemmatizer = WordNetLemmatizer()
    global emojis
    emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
              ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
              ':-@': 'shocked', ':@': 'shocked', ':-$': 'confused', ':\\': 'annoyed',
              ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
              '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
              '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
              ';-)': 'wink', 'O:-)': 'angel', 'O*-)': 'angel', '(:-D': 'gossip', '=^.^=': 'cat'}
    df.Tweet = df.Tweet.apply(remove_noise)
    df.Tweet = [''.join(str(item)) for item in df.Tweet.values]
    x_train, x_test, y_train, y_test = train_test_split(df.Tweet, df.label, test_size=0.3, random_state=2)
    vectorizer = TfidfVectorizer()
    train_vector = vectorizer.fit_transform(x_train)
    test_vector = vectorizer.transform(x_test)
    classifier_linear = svm.SVC(kernel='linear', probability=True)
    classifier_linear.fit(train_vector, y_train)
    prediction_linear = classifier_linear.predict(test_vector)
    report = classification_report(y_test, prediction_linear, output_dict=True)
    print('positive: ', report['1'])
    print('negative: ', report['-1'])
    print('neutral: ', report['0'])
    report_final = classification_report(y_test, prediction_linear, target_names=['-1', '0', '1'])
    print(report_final)
    dump(classifier_linear, '../twitter_classifier_football_v0.joblib')
    dump(vectorizer, '../twitter_vectorizer_football_v0.joblib')


def remove_noise(text):
    clean_text = []
    for token in text:
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)
        token = re.sub('@[^\s]+', ' USER', token)
        for emoji in emojis.keys():
            token = token.replace(emoji, emojis[emoji])
        token = lemmatizer.lemmatize(token.strip(), get_simple_pos(token))
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            clean_text.append(token.lower())
    return clean_text


def get_simple_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


if __name__ == "__main__":
    main()
