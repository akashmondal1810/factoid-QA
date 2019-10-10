data_path = '../data'
import os

directory = os.fsencode(data_path)
fileName_List = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".txt"):
        print(os.path.join(data_path, filename))
        fileName_List.append(os.path.join(data_path, filename))
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


df.to_csv('sentence_with_lbl.csv', index=False)
print("csv file generated from txt file")
