import gensim
import warnings
import string
import collections
import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from many_stop_words import get_stop_words
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from nltk import FreqDist
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
from textblob import TextBlob
import json
import re
from gensim.summarization import keywords
from gensim.summarization import summarize
from gensim import corpora, models
from operator import itemgetter
import csv
from constants import *
from os import path
from scipy.misc import imread
import random


STOP_WORDS = []


def load_stop_words():
    '''
    Appends english stopwords from nltk, many_stop_words,
    custom stopwords from custom_stopwords.txt, and specific stopwords to
    a list STOP_WORDS
    '''
    global STOP_WORDS
    STOP_WORDS = list(get_stop_words('en'))  # About 900 stopwords
    nltk_words = list(stopwords.words('english'))  # About 150 stopwords
    custom_stop_words = list(line.strip() for line in open('custom_stopwords.txt'))
    specific_stop_words = ['came', 'told', 'dont', 'outside', 'okay', 'ok',
                           'oh', 'really', 'never', 'everyone', 'went', 'sat',
                           'well', 'definitely']
    STOP_WORDS.extend(nltk_words)
    STOP_WORDS.extend(custom_stop_words)
    STOP_WORDS.extend(specific_stop_words)


def process_text(text, stem=True):
    '''
    Tokenize text using word_tokenize from nltk,
    removes punctuations and stopwords, and peforms
    lemmatization
    '''
    # text = text.translate(None, string.punctuation)
    tokens = word_tokenize(text)
    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens if t.lower() not in STOP_WORDS]

    return tokens


def cluster_texts(texts, clusters=3):
    '''
    Transform texts to Tf-Idf coordinates and cluster texts using K-Means 
    '''
    vectorizer = TfidfVectorizer(tokenizer=process_text,
                                 stop_words='english',
                                 max_df=0.5,
                                 min_df=0.1,
                                 lowercase=True)
    tfidf_model = vectorizer.fit_transform(texts)

    print(vectorizer.get_feature_names())

    # do clustering
    km_model = KMeans(n_clusters=clusters)
    km_model.fit(tfidf_model)
    clustering = collections.defaultdict(list)
    for idx, label in enumerate(km_model.labels_):
        clustering[label].append(idx)
    return clustering


def get_trimmed_text(text):
    '''
    Function makes use of regular expression extensively 
    to remove urls, punctuations, words with numbers, and unicode
    white spaces
    '''
    
    text = re.sub(r"http\S+", "", text)  # remove urls
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuations
    text = re.sub(r'\w*\d\w*', '', text).strip()  # remove words with numbers in them
    text = re.sub(r"\s+", " ", text, flags=re.UNICODE)  # remove unicode white spaces
    return text


if __name__ == "__main__":

    load_stop_words()

    dataset = None
    with open("review_Lolos.json") as reviews:
        dataset = json.loads(reviews.read())

    if not dataset:
        print("No data found")
        raise ValueError("No data in file")

    review_texts = []
    review_count = 0

    stemmed_texts = []

    for review in dataset:

        if int(review[RATING_POS]) > MIN_RATING_THRESHOLD:
            # skip any review with rating grater than MIN_RATING_THRESHOLD
            continue

        blob = TextBlob(review[REVIEW_POS])

        # check language of review text, translate to default language (english) if required
        language = blob.detect_language()

        if language != DEFAULT_LANG_SYMBOL:
            blob = blob.translate(to=DEFAULT_LANG_SYMBOL)

        # save in string format
        text = str(blob)

        # trim urls, numbers, etc from text
        text = get_trimmed_text(text)
        review_texts.append(text)

        review_count += 1

        if review_count >= DATASET_MAX_SIZE:
            break

    print(len(review_texts))

    clusters = dict(cluster_texts(review_texts, CLUSTER_SIZE))

    problem_list = []
    for cluster_no in range(0, CLUSTER_SIZE):
        problems_per_cluster = ['Cluster %d' % cluster_no]
        text = ""
        # print "CLUSTER %d" % cluster_no,
        for review_number in clusters[cluster_no]:
            text = text + " " + "".join(review_texts[review_number])

        # print text
        noun_phrases = TextBlob(text).noun_phrases
        phrase_count = {}
        for phrase in set(noun_phrases):
            phrase = ' '.join(w for w in phrase.split() if w.lower() not in STOP_WORDS)
            if not phrase or len(phrase) < 2:
                continue

            sentiment = TextBlob(phrase).sentiment

            # ignore phrases which are either neutral or positive, as we are finding problems here
            if (sentiment.polarity == 0.0 and sentiment.subjectivity == 0.0) or sentiment.polarity > 0.4 or (
                            sentiment.polarity > 0.1 and sentiment.subjectivity > 0.8):
                continue

            count = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(phrase), text))
            phrase_count[phrase] = count
        insights = sorted(phrase_count.items(), key=itemgetter(1), reverse=True)[:20]
        for insight in insights:
            problems_per_cluster.append(insight[0])
        problem_list.append(problems_per_cluster)

    csv_file = csv.writer(open('data_size_%d_clusters_%d.csv' % (DATASET_MAX_SIZE, CLUSTER_SIZE), 'w'))
    for row in zip(*problem_list):
        csv_file.writerow(row)
        # for phrase in row:
        # print phrase, "\t\t\t\t\t",
        # print
