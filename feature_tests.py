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

#intiate stemmer, define list of stopwords, definte translation for punctuation removal
ps = SnowballStemmer("english")
#ps = PorterStemmer()
stop_words = set(nltk.corpus.stopwords.words('english'))
punct_remove = string.punctuation.maketrans('', '', string.punctuation)


#intiate lists
neg_list = list()
pos_list = list()
#negTest = list()
#posTest = list()

#open negative training sets and add them as strings onto negative list
for filename in os.listdir('./train/neg'):
    file = open('./train/neg/' + filename, encoding="utf8")
    neg_list.append(file.read())
    file.close()
#open positive training sets and add them as strings onto positive list
for filename in os.listdir('./train/pos'):
    file = open('./train/pos/' + filename, encoding="utf8")
    pos_list.append(file.read())
    file.close()


def word_info_rank(hash1, hash2):
    ranking = {}
    for hash in hash1 not in stop_words:
        if hash in hash2:
            ranking[hash] = max(hash1[hash]*(hash1[hash] + 1)/(hash2[hash] + 2), hash2[hash]*(hash2[hash] + 1)/(hash1[hash] + 2))
        else:
            ranking[hash] = hash1[hash]*(hash1[hash] + 1)/2
    for hash in hash2 not in stop_words:
        if not hash in hash1:
            ranking[hash] = hash2[hash]*(hash2[hash] + 1)/2
    return ranking


def threshold(hash1, hash2, threshold):
    for key in hash1:
        if hash1[key] < 100:
            hash1.pop(key)
    for key in hash2:
        if hash2[key] < 100:
            hash2.pop(key)
    return [hash1, hash2]


#generates a list of words from the frequencies positive and negative frequency lists
def word_ranking(hash1, hash2):
    polarity = {}
    for key in hash1:
        if not key in stop_words:
            polarity[key] = hash1[key]
    for key in hash2:
        if not key in stop_words:
            if key in hash1:
                polarity[key] = abs(hash1[key] - hash2[key])
            else:
                polarity[key] = hash2[key]
    return polarity


def top_words_balanced(hash1, hash2, n):
    topwords = {}
    ranking = word_ranking(hash1, hash2)
    maxN = math.floor(n/2)
    maxP = math.floor(n/2)
    while len(topwords) < n:
        top = max(ranking, key = ranking.get)
        skew = 0
        if top in hash1:
            if top in hash2:
                if hash1[top] > hash2[top]:
                    skew = 0
                else:
                    skew = 1
        if skew == 0 and maxN > 0:
            topwords[top] = ranking[top]
            maxN -= 1
        if skew == 1 and maxP > 0:
            topwords[top] = ranking[top]
            maxP -= 1
        ranking.pop(top)
    return topwords


#extracts the n top words from the list of words by ranking
def top_words(hash, n):
    topWords = {}
    n = min(len(hash), n)
    for i in range(n):
        #topWords.append(hash.pop(max(hash, key = hash.get)))
        top = max(hash, key = hash.get)
        topWords[top] = hash[top]
        hash.pop(top)
    return topWords


#extracts the n min words from the list of words by ranking
def min_words(hash, n):
    topWords = {}
    n = min(len(hash), n)
    for i in range(n):
        #topWords.append(hash.pop(max(hash, key = hash.get)))
        top = min(hash, key = hash.get)
        topWords[top] = hash[top]
        hash.pop(top)
    return topWords


#removes punctuation, < > strings, contractions
def text_process(text):
    #remove line breaks, indenting, punctuation, contractions
    text = re.sub("<.*>", ' ', text)
    text = re.sub("n't", ' not', text)
    text = re.sub("'ve", ' have', text)
    text = text.translate(punct_remove).lower()
    return text


def hash2matdict(hashtable):
    translator = {}
    inverse = {}
    i = 0
    for hash in hashtable:
        translator[int(i)] = hash
        inverse[hash] = i
        i += 1
    print(translator)
    return [translator, inverse]


def compute_adj(docs, words):
    n = len(words)
    data = np.zeros([n, n])
    for text in docs:
        text = text_process(text)
        tokens = wt(text)
        stemtokens = [ps.stem(word) for word in tokens if not word in stop_words]
#        stemtokens += [tokens[i] + ps.stem(tokens[i + 1]) for i in range(len(tokens) - 1) if tokens[i] == 'not' and not tokens[i + 1] in stop_words]
        for i in range(0, n):
            if words[i] in stemtokens:
                data[i, i] += 1
                for j in range(i):
                    if words[j] in stemtokens:
                        data[i, j] += 1
    for i in range(n):
        for j in range(i):
            data[j, i] = data[i, j]
    data /= len(docs)
    return data



def csv2hash(csvname):
    with open(csvname, 'r', encoding="utf8") as hashfile:
        text = hashfile.read()
    text = text.replace(',', ' ')
    tokens = wt(text)
    tokens = tokens[2:len(tokens)]
    hashsize = int(len(tokens)/2)
    hashmap = {}
    for i in range(hashsize):
        hashmap[tokens[2*i]] = float(tokens[2*i + 1])
    return hashmap


def hash2csv(hashlist, file, col1, col2):
    with open(file, 'w', encoding="utf8") as csv_file:
        fieldnames = [col1, col2]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for key in hashlist.keys():
        writer.writerow({'Word': key,'Frequency': bestfeat[key]})



def word_scoring(adjmat, translator):
    scores = {}
    for feat1 in translator:
        entropy = 0
        key = translator[feat1]
        for feat2 in translator:
            if not feat1 == feat2 and adjmat[feat1, feat2] > 0:
                entropy += adjmat[feat1, feat2]*math.log(adjmat[feat1,feat2]/((adjmat[feat1, feat1]*adjmat[feat2, feat2])), 2)
        scores[key] = entropy
    return scores




neg_freq_list = csv2hash('negwordcounts.csv')
pos_freq_list = csv2hash('poswordcounts.csv')

[neg, pos] = threshold(neg_freq_list, pos_freq_list, 100)
hash2csv(neg, 'negwordcounts.csv', 'Word', 'Frequency')
hash2csv(pos, 'poswordcounts.csv', 'Word', 'Frequency')
print(len(neg))
print(len(pos))

#train = neg_list[6250:len(neg_list)] + pos_list[6250:len(pos_list)]

#bestfeat = top_words(word_ranking(neg_freq_list, pos_freq_list), 2000)
#bestfeat = top_words_balanced(neg_freq_list, pos_freq_list, 2000)

'''
top = top_words(ranking, 1000)
[translator, inverse] = hash2matdict(top)

adjmat = compute_adj(train, translator)
scores = word_scoring(adjmat, translator)
bestfeat = top_words(scores, 500)
newtrans = {}
for key in bestfeat:
    index = int(inverse[key])
    newtrans[index] = key
scores = word_scoring(adjmat, newtrans)
bestfeat = top_words(scores, 250)
'''


