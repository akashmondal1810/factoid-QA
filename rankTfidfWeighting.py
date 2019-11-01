from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
import json

import nltk
import os
import string
import numpy as np
import copy
import pandas as pd
import pickle
import re
import math

# %load_ext autotime

title = "data"
alpha = 0.3
folders = [x[0] for x in os.walk(str(os.getcwd())+'/'+title+'/')]
folders[0] = folders[0][:len(folders[0])-1]

directory = os.fsencode(folders[0])
dataset = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".txt"):
        #print(os.path.join(folders[0], filename))
        dataset.append(os.path.join(folders[0], filename))
        continue
    else:
        continue
N = len (dataset)


def print_doc(id):
    print(dataset[id])
    file = open(dataset[id], 'r', encoding='cp1250')
    text = file.read().strip()
    file.close()
    print(text)

def convert_lower_case(data):
    return np.char.lower(data)

def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text

def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data

def remove_apostrophe(data):
    return np.char.replace(data, "'", "")

def stemming(data):
    stemmer= PorterStemmer()
    
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text

def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data) 
    data = remove_apostrophe(data)
    data = remove_stop_words(data)
    data = stemming(data)
    return data


def matching_score(k, query):
    with open("pass_ret/data_dfidf.json","r") as f:
        tf_idf = dict([tuple((tuple(x[0]), x[1])) for x in json.loads(f.read())])
    preprocessed_query = preprocess(query)
    tokens = word_tokenize(str(preprocessed_query))

    #print("Matching Score")
    #print("\nQuery:", query)
    #print("")
    #print(tokens)
    
    query_weights = {}

    for key in tf_idf:
        
        if key[1] in tokens:
            try:
                query_weights[key[0]] += tf_idf[key]
            except:
                query_weights[key[0]] = tf_idf[key]
    
    query_weights = sorted(query_weights.items(), key=lambda x: x[1], reverse=True)

    #print("")
    
    l = []
    d= []
    for i in query_weights[:10]:
        l.append(i[0])
        d.append(dataset[i[0]])
    
    return d
    

#matching_score(10, "Which Bengali sweet consists of paneer balls dipped in sugar syrup")

