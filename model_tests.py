import csv
import numpy as np
import math
import matplotlib as plt
import os
import string
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import *
from nltk.tokenize import sent_tokenize as st, word_tokenize as wt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestClassifier as RandomForest
from nltk.corpus import wordnet

# Use SnowballStemmer to stem input comments.
ps = SnowballStemmer("english")
# Use nltk's predefined stopword list as our stop_words set.
stop_words = set(nltk.corpus.stopwords.words('english'))

# Remove all occurrences of punctuation with this function.
punct_remove = string.punctuation.maketrans('', '', string.punctuation)

# Initiate list
x_train = list()
y_train = list()
x_test = list()
y_test = list()
pos_list = list()

# Open Negative Training Data and append entries to list.
for filename in os.listdir('./train/neg'):
    file = open('./train/neg/' + filename, encoding="utf8")
    x_train.append(file.read())
    y_train.append(0)
    file.close()

# Open Positive Training Data and append entries to list.
for filename in os.listdir('./train/pos'):
    file = open('./train/pos/' + filename, encoding="utf8")
    x_train.append(file.read())
    y_train.append(1)
    file.close()


# returns total counts of each word across all docs
def stem_words(data):
    def feature_tokens(tokens):
        stemtokens = list()
        for i in range(len(tokens)):
            if tokens[i] == 'not':
                i += 1
                continue
            if tokens[i] not in stop_words and not tokens[i].endswith("i"):
                stemmed = ps.stem(tokens[i])
                if len(stemmed) > 2:
                    stemtokens.append(stemmed)
        return stemtokens
    # initiate list for counting word frequencies in the list of documents
    new_train = list()
    for rawtext in data:
        # remove line breaks, indenting, punctuation, contractions
        text = processText(rawtext)

        # adds all stems that aren't stopwords
        tokens = wt(text)
        stemtokens = feature_tokens(tokens)
        new_train.append(' '.join(stemtokens))
#    print(new_train)
    return new_train


# remove line breaks, indenting, punctuation, contractions
def processText(text):
    #text = re.sub("<.*>", ' ', text)
    text = re.sub("n't", ' not', text)
    text = re.sub("'ve", ' have', text)
    text = text.translate(punct_remove).lower()
    return text


#new_train = stem_words(x_train)

X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.85, test_size=0.15)
for i in range(len(X_train)):
    x_train[i] = processText(x_train[i])
#print(X_train)
#print(y_train)

'''
for c in [0.01, 0.05, 0.25, 0.5, 0.6, 0.75, 1]:
    pclf = Pipeline([
        ('vect', CountVectorizer(binary=True)),
#        ('tfidf', TfidfTransformer()),
        ('norm', Normalizer()),
        ('clf', LogisticRegression(C=c)),
    ])

    pclf.fit(X_train, y_train)
    y_pred = pclf.predict(X_test)

    print("C = %s"%(c))
    print(metrics.classification_report(y_test, y_pred))

'''
for value in [1]:
    pclf = Pipeline([
        ('vect', CountVectorizer()),
#        ('tfidf', TfidfTransformer()),
#        ('norm', Normalizer()),
        ('clf', RandomForest(n_estimators = 100, max_depth = 500)),
    ])

    pclf.fit(X_train, y_train)
    y_pred = pclf.predict(X_test)

    print("C = %s"%(value))
    print(metrics.classification_report(y_test, y_pred))
