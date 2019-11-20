from bert import QA
model = QA('model')

def bertAns(q, doc):
    return model.predict(doc,q)
