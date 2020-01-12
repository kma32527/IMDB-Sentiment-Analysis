
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

# intiate stemmer, define list of stopwords, definte translation for punctuation removal
ps = SnowballStemmer("english")
# ps = PorterStemmer()
stop_words = set(nltk.corpus.stopwords.words('english'))
punct_remove = string.punctuation.maketrans('', '', string.punctuation)

# intiate lists
neg_list = list()
pos_list = list()

# open negative training sets and add them as strings onto negative list
for filename in os.listdir('./train/neg'):
    file = open('./train/neg/' + filename, encoding="utf8")
    neg_list.append(file.read())
    file.close()
# open positive training sets and add them as strings onto positive list
for filename in os.listdir('./train/pos'):
    file = open('./train/pos/' + filename, encoding="utf8")
    pos_list.append(file.read())
    file.close()

# attempt at converting the test data into lists (does not work)
'''
with open('sample_submission.csv') as scores:
    txt = scores.read()
    txt = txt.replace(',', ' ')
    tokens = wt(txt)
    tokens = tokens[2: len(tokens)]
    for i in range(int(len(tokens)/2)):
        with open('./test/' + str(i) + '.txt', encoding="utf8") as example:
            if int(tokens[2*i + 1]) == 0:
                negTest.append(example.read())
            else:
                posTest.append(example.read())
'''


# generates a list of words from the positive and negative frequency lists
def word_ranking(hash1, hash2):
    polarity = {}
    for hash in hash1:
        polarity[hash] = math.log(12500 / hash1[hash]) * hash1[hash]
    for hash in hash2:
        if hash in hash1:
            polarity[hash] = math.log(12500 / (hash1[hash] + hash2[hash])) * max(hash1[hash], hash2[hash])
        else:
            polarity[hash] = math.log(12500 / hash2[hash]) * hash2[hash]
    return polarity


# extracts the n top words from the list of words by ranking
def top_words(hash, n):
    topWords = list()
    n = min(len(hash), n)
    for i in range(n):
        # topWords.append(hash.pop(max(hash, key = hash.get)))
        top = max(hash, key=hash.get)
        if i == n - 1:
            print(hash[top])
        topWords.append(top)
        hash.pop(top)
    print(topWords)
    return topWords


# removes punctuation, < > strings, contractions
def text_process(text):
    # remove line breaks, indenting, punctuation, contractions
    text = re.sub("<.*>", ' ', text)
    text = re.sub("n't", ' not', text)
    text = re.sub("'ve", ' have', text)
    text = text.translate(punct_remove).lower()
    return text


def feature_tokens(tokens):
    stemtokens = list()
    for i in range(len(tokens)):
        if not tokens[i] in stop_words:
            if i > 0 and tokens[i - 1] == 'not':
                stemtokens.append(tokens[i - 1] + ps.stem(tokens[i]))
            else:
                stemtokens.append(ps.stem(tokens[i]))
    return stemtokens


# takes each example and outputs a matrix
# [i, j] = 1 if example i has feature j, 0 otherwise
def extract_data(examples, features):
    data = np.zeros([len(examples), len(features)])
    for entry in range(len(data)):
        text = text_process(examples[entry])
        tokens = wt(text)
        stemtokens = feature_tokens(tokens)
        for i in range(len(features)):
            if features[i] in stemtokens:
                data[entry, i] = 1
    return data


# trains weights and outputs a nx2 matrix where n is number of features
# weights[i, j] = conditional probability of i given score j
def naive_bayes(dataN, dataP):
    condP = np.zeros([len(dataP[0]), 2])
    # compute probability of each feature with laplace smoothing
    for i in range(len(condP)):
        totalN = 0
        totalP = 0
        for entry in dataN:
            totalN += entry[i]
        for entry in dataP:
            totalP += entry[i]
        condP[i, 0] = (totalN + 1) / (len(dataN) + 2)
        condP[i, 1] = (totalP + 1) / (len(dataP) + 2)
    return condP


# Converts CSV to Hash Table by Word-Tokenizing CSV files and alternatively assigning tokens
# as keys then values.
def csvtohash(csvname):
    with open(csvname, 'r', encoding="utf8") as hashfile:
        text = hashfile.read()
    text = text.replace(',', ' ')
    tokens = wt(text)
    tokens = tokens[2:len(tokens)]
    hashsize = int(len(tokens) / 2)
    hashmap = {}
    for i in range(hashsize):
        hashmap[tokens[2 * i]] = float(tokens[2 * i + 1])
    return hashmap


# uses conditional probability weights and computes predictions
def check(data, weights, score):
    inclusionmat = np.ones([len(data), len(data[0])])
    nhasx = np.log(weights[:, 0] / weights[:, 1])
    phasx = np.log(weights[:, 1] / weights[:, 0])
    nnox = np.log((1 - weights[:, 0]) / (1 - weights[:, 1]))
    pnox = np.log((1 - weights[:, 1]) / (1 - weights[:, 0]))
    probN = np.matmul(inclusionmat, nnox) + np.matmul(data, nhasx - nnox)
    probP = np.matmul(inclusionmat, pnox) + np.matmul(data, phasx - pnox)
    predictions = probP - probN
    for i in range(len(predictions)):
        if predictions[i] < 0:
            predictions[i] = 1 - score
        else:
            predictions[i] = score
    return np.sum(predictions) / len(predictions)


# converts csv to hash table for positive and negative lists
neg_freq_list = csvtohash('negwordcounts.csv')
pos_freq_list = csvtohash('poswordcounts.csv')

# extract features
ranking = word_ranking(neg_freq_list, pos_freq_list)
features = top_words(ranking, 1000)
# featuresdoc = csvtohash('bestfeatures.csv')
# features = [key for key in featuresdoc]
print(features)

# sets training and validation sets
[neg_train, pos_train] = [neg_list[6250:len(neg_list)], pos_list[6250:len(pos_list)]]
[neg_valid, pos_valid] = [neg_list[:6250], pos_list[:6250]]
[neg_train_data, pos_train_data] = [extract_data(neg_train, features), extract_data(pos_train, features)]
[neg_val_data, pos_val_data] = [extract_data(neg_valid, features), extract_data(pos_valid, features)]

weights = naive_bayes(neg_train_data, pos_train_data)

neg_train_error = check(neg_train_data, weights, 0)
pos_train_error = check(pos_train_data, weights, 1)
neg_val_error = check(neg_val_data, weights, 0)
pos_val_error = check(pos_val_data, weights, 1)
# print everything
print("Neg Acc on training: " + str(neg_train_error))
print("Pos Acc on training: " + str(pos_train_error))
print("Total Acc on training: " + str((neg_train_error + pos_train_error) / 2))
print("Neg Acc on validation: " + str(neg_val_error))
print("Pos Acc on validation: " + str(pos_val_error))
print("Total Acc on validation: " + str((neg_val_error + pos_val_error) / 2))
