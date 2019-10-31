from rankTfidfWeighting import matching_score
from qProcessing_nltk import QpreProcessing
from rankByclassify import docRank_by_classfication
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import os
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from word2vec_repo.DocSim import DocSim
from nltk.corpus import stopwords

#some of the code are from https://stackoverflow.com/a/8897648 and https://github.com/v1shwa/document-similarity/blob/master/example.py

model_path = 'word2vec_repo/model.bin'
stopwords = stopwords.words('english')

model = KeyedVectors.load(model_path)
ds = DocSim(model, stopwords=stopwords)

#fetch top 3 doc using tf-idf weighting method
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
        tl = [str(state[len(state)-1]).lower()[:-4] +", " +line for line in tl]
        answer_doc+=tl

    answer_doc.append(qu)
    return answer_doc

#fetch ans doc using classification method
def get_doc_tc(qu):
    final_docs = docRank_by_classfication(qu)
    final_doc = final_docs[0]

    answer_doc = []
    file = open(final_doc, 'r', encoding='cp1250')
    text = file.read().strip()
    file.close()
    text=text.replace('\n', ' ')
    tl = nltk.sent_tokenize(text)
    state = str(final_doc).split('/')
    tl = [str(state[len(state)-1]).lower()[:-4] +", " +line for line in tl]
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
    try:
        return sim_scores[1]['doc']
    except:
        return "unable to fetch answer/answer not present"


'''
q = str(input("Question: "))
ans_lists = get_doc(q)
print("tf-idf: ", get_answer(q, ans_lists))
print("----------------------------------")
print("w2v: ", get_ans_w2v(q, ans_lists))
'''
import pandas as pd
import time
df = pd.read_excel(r"documents/Question.xlsx",sheetname='Sheet1')
df["ans_tfIdf"]= [get_answer(q, get_doc(q)) for q in df['Question']]
df["ans_w2v"]= [get_ans_w2v(q, get_doc(q)) for q in df['Question']]
df.to_excel('documents/Question_&_ans_13k.xlsx', sheet_name='Sheet1')

'''
qp = QpreProcessing()


time_qp = []
formated_q = []
for q in df['Question']:
    start_time = time.time()
    f = qp.qType_words(q)
    time_qp.append((time.time() - start_time))
    formated_q.append(f[len(f)-1])

df["exe_time_qp"]= time_qp
df["formated_question"] = formated_q

time_pr = []
final_doc = []
for q in df['formated_question']:
    start_time = time.time()
    f = get_doc(q)
    time_pr.append((time.time() - start_time))
    final_doc.append(f)
    
df["exe_time_pr"]= time_pr
df["ans_doc"] = final_doc


time_ap = []
answer = []
for q, ans_d in zip(df['formated_question'], df["ans_doc"]):
    start_time = time.time()
    f = get_answer(q, ans_d)
    time_ap.append((time.time() - start_time))
    answer.append(f)
    

df["exe_time_ap_tf"]= time_ap
df["answer_tf"]= answer

time_ap1 = []
answer1 = []
for q, ans_d in zip(df['formated_question'], df["ans_doc"]):
    start_time = time.time()
    f = get_ans_w2v(q, ans_d)
    time_ap1.append((time.time() - start_time))
    answer1.append(f)
    

df["exe_time_ap_w2v"]= time_ap1
df["answer_w2v"]= answer1


df["ans_tfIdf"]= [get_answer(q, get_doc(q)) for q in df['Question']]

time_tf = []
for q in df['Question']:
    start_time = time.time()
    get_answer(q, get_doc(q))
    time_tf.append((time.time() - start_time))
df["total_time_tfIdf"]= time_tf

df["ans_w2v"]= [get_ans_w2v(q, get_doc(q)) for q in df['Question']]

time_w2v = []
for q in df['Question']:
    start_time = time.time()
    get_ans_w2v(q, get_doc(q))
    time_w2v.append((time.time() - start_time))
df["total_time_w2v"]= time_w2v

df.to_excel('documents/Question_&_ans_anlysis_tf.xlsx', sheet_name='Sheet1')
'''
