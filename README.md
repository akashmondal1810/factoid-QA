# factoid-QA

## Setup:
- Install dependencies `pip install -r dependencies.txt`
- Download the BERT Pretrained model trained on [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) from [here](https://www.dropbox.com/s/8jnulb2l4v7ikir/model.zip) and extract it inside `BERTap` folder
- Download the `glove.6B.300d.txt` and extract it inside `word2vec_repo/glove6b` folder. Also each corpus need to start with a line containing the vocab size and the vector size in that order. So in our case add this line "400000 300" as the first line of `glove.6B.300d.txt`.
- Download nltk stopwords:
```python
    import nltk
    nltk.download('stopwords')
```
- Run `processed_data/txt_to_csv.py` to generate the processed dataframe
- Run `pass_ret/txt_to_tfidf_dict.py` to generate the tf-idf values over the corpus
- Run `word2vec_repo/gen_w2v_modal.py` to generate the word2vec model
### Default directory structure
```
QAsys
├── data (or $QA_Knowledge_DATA)
|   └── <unique/non overlapping doc name>.txt
├── word2vec_repo   
│   ├── glove6b
│   |   └── glove.6B.300d.txt
│   └── domain2vec.bin (or $TRAINED_W2V_MODEL)
├── pass_ret
│   └── data_tfidf.json
└── BERTap
        └── model
            ├──bert_config.json
            └── pytorch_model.bin
                (or $TRAINED_SQuAD_MODEL)
```

## Execute:
A basic User Interface using Python's Tkinter library is provided .
- `python UI_app.py` for answer
