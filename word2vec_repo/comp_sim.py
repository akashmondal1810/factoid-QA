from gensim.models.keyedvectors import KeyedVectors
from DocSim import DocSim
from nltk.corpus import stopwords

model_path = 'model.bin'
stopwords = stopwords.words('english')

model = KeyedVectors.load(model_path)
ds = DocSim(model, stopwords=stopwords)

source_doc = "what are the ingredients of litti chokha"
target_docs = ['Bihari cuisine may include litti chokha', 'litti chokha, a baked salted wheat-flour cake filled with sattu', "Dalpuri is another popular dish in Bihar"]

sim_scores = ds.calculate_similarity(source_doc, target_docs)

print(sim_scores)
