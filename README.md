# factoid-QA

## Setup:
- Install dependencies `pip install -r dependencies.txt`
- Download the BERT Pretrained model trained on [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) from [here](https://www.dropbox.com/s/8jnulb2l4v7ikir/model.zip) and extract it inside `BERTap` folder
- Download nltk stopwords:
```python
    import nltk
    nltk.download('stopwords')
```
- Run `cd processed_data` and `python txt_to_csv.py` to generate the processed dataframe and then `cd ..`
- Run `cd pass_ret` and `python txt_to_tfidf_dict.py` to generate the tf-idf dictionary and then `cd ..`
- Run `cd word2vec_repo` and `python gen_w2v_modal.py` to generate the word2vec model and then `cd ..`

## Execute:
- `python UI_app.py` for answer
- Sample questions - [here](https://github.com/akashmondal1810/factoid-QA/blob/master/documents/Question.csv)
