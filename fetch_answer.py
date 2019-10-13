from fetch_final_docs import matching_score
from qProcessing_nltk import QpreProcessing
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import os
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from word2vec_repo.DocSim import DocSim
from nltk.corpus import stopwords

#some of the code are from https://stackoverflow.com/a/8897648 

model_path = 'word2vec_repo/model.bin'
stopwords = stopwords.words('english')

model = KeyedVectors.load(model_path)
ds = DocSim(model, stopwords=stopwords)


def get_doc(qu):
    final_docs = matching_score(10, qu)
    #print(final_docs)
    #print(final_docs)
    nd= len(final_docs)
    if len(final_docs)>3:
        nd = 3
    answer_doc = []
    for i in range(nd):
        file = open(final_docs[i], 'r', encoding='cp1250')
        text = file.read().strip()
        file.close()
        text=text.replace('\n', ' ')
        tl = nltk.sent_tokenize(text)
        state = str(final_docs[i]).split('/')
        tl = [str(state[len(state)-1])[:-4] +". " +line for line in tl]
        answer_doc+=tl

    answer_doc.append(qu)
    return answer_doc

def get_answer(qu, answer_sent):


    vect = TfidfVectorizer(min_df=1, stop_words="english")                                                                                                                                                                                                   
    tfidf = vect.fit_transform(answer_sent)                                                                                                                                                                                                                       
    pairwise_similarity = tfidf * tfidf.T      

    arr = pairwise_similarity.toarray()     
    np.fill_diagonal(arr, np.nan)                                                                                                                                                                                                                            

    input_doc = qu                                                                                                                                                                                                 
    input_idx = answer_sent.index(input_doc)                                                                                                                                                                                                                      

    result_idx = np.nanargmax(arr[input_idx])                                                                                                                                                                                                                
    return answer_sent[result_idx] 

def get_ans_w2v(qu, answer_sent):

    source_doc = qu
    target_docs = answer_sent

    sim_scores = ds.calculate_similarity(source_doc, target_docs)
    sorted(sim_scores, key = lambda i: i['score'])
    return sim_scores[1]['doc']

'''
q = str(input("Question: "))
ans_lists = get_doc(q)
print("tf-idf: ", get_answer(q, ans_lists))
print("----------------------------------")
print("w2v: ", get_ans_w2v(q, ans_lists))

import pandas as pd
import time
df = pd.read_csv(r"documents/Question.csv",encoding='latin-1')
df["ans_tfIdf"]= [get_answer(q, get_doc(q)) for q in df['Question']]

time_tf = []
for q in df['Question']:
    start_time = time.time()
    get_answer(q, get_doc(q))
    time_tf.append((time.time() - start_time))
df["exe_time_tfIdf"]= time_tf

df["ans_w2v"]= [get_ans_w2v(q, get_doc(q)) for q in df['Question']]

time_w2v = []
for q in df['Question']:
    start_time = time.time()
    get_ans_w2v(q, get_doc(q))
    time_w2v.append((time.time() - start_time))
df["exe_time_w2v"]= time_w2v

df.to_csv('documents/Question_&_ans_genrtd.csv', index=False)
'''
