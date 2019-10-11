# factoid-QA

## Setup:
- Install dependencies `pip install -r dependencies.txt`
- Download nltk stopwords:
```python
    import nltk
    nltk.download('stopwords')
```
- Run `cd pass_ret`
- Run `python txt_to_tfidf_dict.py` to generate the tf-idf dictionary

## Execute:
- `python UI_app.py` for answer
- Sample questions - [here](https://github.com/akashmondal1810/factoid-QA/blob/master/documents/Question.csv)
