import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.tree import Tree
from qType import processquestion
import json
stopword = set(stopwords.words('english'))
stopword.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])


class QpreProcessing:
    
    def __init__(self):
        pass

    
    def keywords(self, sentence):
        '''removes stop words too'''
        doc = [i for i in word_tokenize(sentence) if i not in stopword]
        return doc
    
    # NLTK POS and NER taggers   
    def nltk_tagger(self, token_text):
        tagged_words = nltk.pos_tag(token_text)
        ne_tagged = nltk.ne_chunk(tagged_words)
        return(ne_tagged)
    
    # Parse named entities from tree
    def structure_ne(self, ne_tree):
        ne = []
        for subtree in ne_tree:
            if type(subtree) == Tree: # If subtree is a noun chunk, i.e. NE != "O"
                ne_label = subtree.label()
                ne_string = " ".join([token for token, pos in subtree.leaves()])
                ne.append((ne_string, ne_label))
        return ne

    def pos(self, sentence):
        sent = nltk.word_tokenize(sentence)
        sent = nltk.pos_tag(sent)
        return sent
    
    def ner(self, sentence):
        return self.structure_ne(self.nltk_tagger(word_tokenize(sentence)))
    
    def qType_words(self, sentence):
        qwords  = word_tokenize(sentence)
        (type, target) = processquestion(qwords)
        target = str(" ".join(map(str, target)))
        if type=='YESNO':
            return [type, target]
        else:
            qDetail = ['FACTOID']
            qDetail=qDetail+[type, target]
            return qDetail
    
if __name__ == '__main__':
    sNLP = QpreProcessing()
    text1 = str(input('Question: '))
    text = text1.replace('?','')
    qpResult = {'Question':text1}
    
    qpResult['tokenize'] = sNLP.keywords(text)
    qpResult['ner'] = sNLP.ner(text)
    qpResult['pos'] = sNLP.pos(text)
    qpResult['qdetail'] = sNLP.qType_words(text)

    with open('qpResult.json', 'w') as fp:
        json.dump(qpResult, fp)
    
    print(qpResult)
