import os

title = "../data"
folders = [x[0] for x in os.walk(str(os.getcwd())+'/'+title+'/')]
folders[0] = folders[0][:len(folders[0])-1]

directory = os.fsencode(folders[0])
fileName_List = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".txt"):
        print(os.path.join(folders[0], filename))
        fileName_List.append(os.path.join(folders[0], filename))
        continue
    else:
        continue

import nltk
import pandas as pd
columns = ['sentence', 'lable']
df = pd.DataFrame(columns=columns) #empty dataframe

for txt_file in fileName_List:
    with open(txt_file, 'r') as in_file:
        text = in_file.read()
        sents = nltk.sent_tokenize(text)

    #print(sents)
    labl = [str(txt_file) for i in range(len(sents))]
    txt_df = pd.DataFrame({'sentence': sents, 'lable': labl })
    #print(txt_df)

    df = df.append(txt_df, ignore_index=True)

from sklearn.utils import shuffle
df = shuffle(df)
df.to_excel('sentence_with_lbl.xlsx', sheet_name='Sheet1')
#df.to_csv('sentence_with_lbl.csv', index=False)
print("xlsx file generated from txt file")
