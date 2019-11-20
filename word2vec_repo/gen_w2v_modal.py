import logging
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim.models.word2vec as w2v
import multiprocessing
import nltk
import os
import numpy as np

from gensim.models import KeyedVectors
glove_path = "glove6b/glove.6B.300d.txt"


title = "../data"
folders = [x[0] for x in os.walk(str(os.getcwd())+'/'+title+'/')]
folders[0] = folders[0][:len(folders[0])-1]

directory = os.fsencode(folders[0])
print('Data Loading from'+str(directory))

fileName_List = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".txt"):
        #print(os.path.join(folders[0], filename))
        fileName_List.append(os.path.join(folders[0], filename))
        continue
    else:
        continue

sents = []
for txt_file in fileName_List:
    with open(txt_file, 'r') as in_file:
        text = in_file.read()
        sents += nltk.sent_tokenize(text)
print('Data Loading completed')

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
    symbols = "!\"#$%&()*+-./:,;'<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
    return data

def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data) 
    data = remove_stop_words(data)

    return data


sents= [preprocess(entry) for entry in sents]

bigger_list=[]
for i in sents:
    li = list(i.split(" "))
    bigger_list.append(li)

print('Data preprocessing completed')

num_features = 300
# Minimum word count threshold.
min_word_count = 1

# Number of threads to run in parallel.
#more workers, faster we train
num_workers = multiprocessing.cpu_count()

# Context window length, yeild best results for our dataset
context_size = 6

# Downsample setting for frequent words.
#tested on 0 - 1e-5 
downsampling = 1e-3
seed = 1

print('Training starting over the weight of glove..')
food2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)
food2vec.build_vocab(bigger_list)

total_examples = food2vec.corpus_count
model = KeyedVectors.load_word2vec_format(glove_path, binary=False)

food2vec.build_vocab([list(model.vocab.keys())], update=True)
food2vec.intersect_word2vec_format(glove_path, binary=False, lockf=1.0)
food2vec.train(bigger_list, total_examples=total_examples, epochs=food2vec.iter)

print('Training completed')
food2vec.save("word2vec.model")
food2vec.save("model.bin")

print("modal saved")
