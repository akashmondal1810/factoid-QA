from nltk.corpus import stopwords
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
import nltk
import os
import numpy as np
import copy
import pickle
import re
import math

df = pd.read_excel(r'processed_data/sentence_with_lbl.xlsx', sheet_name='Sheet1')

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
    symbols = "!\"#$%&()*+-./:,;<=>'?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    return data


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
    data = remove_stop_words(data)
    data = stemming(data)
    return data



df['sentence']= [preprocess(entry) for entry in df['sentence']]
X = df['sentence'] # Collection of documents
y = df['lable']

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=1,
                        ngram_range=(1, 2), 
                        stop_words='english')

fitted_vectorizer = tfidf.fit(X)

def get_fitted_vectorizer():
    return fitted_vectorizer

tfidf_vectorizer_vectors = fitted_vectorizer.transform(X)

model = LinearSVC().fit(tfidf_vectorizer_vectors, y)

def docRank_by_classfication(q):
    return model.predict(fitted_vectorizer.transform([preprocess(q)]))
