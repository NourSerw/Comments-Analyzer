from joblib import load
import nltk


textblob_0 = open("F:\\dev\\Comments-Analyzer\\commentsanalyzer\\twitter_classifier_football_v0.joblib", "rb")
classifier_textblob = load(textblob_0)

textblob_1 = open("F:\\dev\\Comments-Analyzer\\commentsanalyzer\\twitter_vectorizer_football_v0.joblib", "rb")
vector_textblob = load(textblob_1)


local_classifier_0 = open("F:\\dev\\Comments-Analyzer\\commentsanalyzer\\twitter_classifier_football_v1.joblib","rb")
classifier_local = load(local_classifier_0)

local_classifier_1 = open("F:\\dev\\Comments-Analyzer\\commentsanalyzer\\twitter_vectorizer_football_v1.joblib","rb")
vector_local = load(local_classifier_1)

pos_text = "Had so much fun at the festival today!"
neutral_text = "The door to the room."
neg_text = "The service at the diner was the worst I have ever seen."

test_text = [pos_text, neutral_text, neg_text]
stopwords = nltk.corpus.stopwords.words('english')
textBlob_conf = ''
localclf_conf = ''
for i in range(0, len(test_text)):
    vec_0_rev = vector_textblob.transform([test_text[i]])
    classifier_0_rev = classifier_textblob.predict(vec_0_rev)
    classifier_0_prob = classifier_textblob.predict_proba(vec_0_rev)
    if classifier_0_rev == -1:
        textBlob_conf = classifier_0_prob[0][0]
    elif classifier_0_rev == 0:
        textBlob_conf = classifier_0_prob[0][1]
    elif classifier_0_rev == 1:
        textBlob_conf = classifier_0_prob[0][2]

    vec_1_rev = vector_local.transform([test_text[i]])
    classifier_1_rev = classifier_local.predict(vec_1_rev)
    classifier_1_prob = classifier_local.predict_proba(vec_1_rev)

    if classifier_1_rev == -1:
        localclf_conf = classifier_1_prob[0][0]
    elif classifier_1_rev == 0:
        localclf_conf = classifier_1_prob[0][1]
    elif classifier_1_rev == 1:
        localclf_conf = classifier_1_prob[0][2]

    print("TextBlob confidence: ", textBlob_conf)
    print("Local clf confidence: ", localclf_conf)
    print('\n')
