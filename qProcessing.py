from nltk.tag import StanfordNERTagger
from nltk.tag.stanford import StanfordPOSTagger as POS_Tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from qType import processquestion
stopword = set(stopwords.words('english'))
stopword.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])


class QpreProcessing:
    
    def __init__(self):
        self.st_NER_PATH1 = 'stanford-nlp/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz'
        self.st_NER_PATH2 = 'stanford-nlp/stanford-ner/stanford-ner.jar'
        self.st_NER = StanfordNERTagger(self.st_NER_PATH1, self.st_NER_PATH2, encoding='utf-8')

        self.st_POS_PATH1 = 'stanford-nlp/postagger/models/english-bidirectional-distsim.tagger'
        self.st_POS_PATH2 = 'stanford-nlp/postagger/stanford-postagger.jar'
        self.st_POS = POS_Tag(self.st_POS_PATH1, self.st_POS_PATH2)

    
    def keywords(self, sentence):
        '''removes stop words too'''
        doc = [i for i in word_tokenize(sentence) if i not in stopword]
        return doc

    def pos(self, sentence):
        pos_text = self.st_POS.tag(self.keywords(sentence))
        return pos_text
    
    def ner(self, sentence):
        ner_text = self.st_NER.tag(self.keywords(sentence))
        return ner_text
    
    def qType_words(self, sentence):
        qwords  = word_tokenize(sentence)
        (type, target) = processquestion(qwords)
        if type=='YESNO':
            return [type, target]
        else:
            qDetail = ['FACTOID']
            qDetail=qDetail+[type, target]
            return qDetail
    
if __name__ == '__main__':
    sNLP = QpreProcessing()
    text = str(input('Question: '))
    tree1 = sNLP.keywords(text)
    tree2 = sNLP.ner(text)
    tree3 = sNLP.pos(text)
    tree4 = sNLP.qType_words(text)
    print("tokenize:", tree1)
    print("ner:", tree2)
    print("POS:", tree3)
    print("qdetail:", tree4)
