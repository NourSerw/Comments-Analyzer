import re
import string

import nltk
import pandas as pd
from joblib import dump
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def main():
    df = pd.read_csv('/Reddit_Data.csv')
    print("CSV loaded")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    df = df.drop(df[df.clean_comment.isna()].index)
    df.clean_comment = df.apply(lambda row: nltk.word_tokenize(row['clean_comment']), axis=1)
    stop = set(stopwords.words('english'))
    punctuation = list(string.punctuation)
    stop.update(punctuation)
    global stop_words
    stop_words = stopwords.words('english')
    global lemmatizer
    lemmatizer = WordNetLemmatizer()
    df.clean_comment = df.clean_comment.apply(remove_noise)
    df.clean_comment = [''.join(str(item)) for item in df.clean_comment.values]
    x_train, x_test, y_train, y_test = train_test_split(df.clean_comment, df.category, test_size=0.3, random_state=0)
    vectorizer = TfidfVectorizer()
    train_vector = vectorizer.fit_transform(x_train)
    test_vector = vectorizer.transform(x_test)
    classifier_linear = svm.SVC(kernel='linear')
    classifier_linear.fit(train_vector, y_train)
    prediction_linear = classifier_linear.predict(test_vector)
    report = classification_report(y_test, prediction_linear, output_dict=True)
    print('positive: ', report['1'])
    print('negative: ', report['-1'])
    print('neutral: ', report['0'])
    report_final = classification_report(y_test, prediction_linear, target_names=['-1', '0', '1'])
    print(report_final)
    dump(classifier_linear, '../reddit_classifier_FINAL_LOCAL.joblib')
    dump(vectorizer, '../TfidfVectorizer_vectorizer_LOCAL.joblib')

def remove_noise(text):
    clean_text = []
    for token in text:
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)
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
