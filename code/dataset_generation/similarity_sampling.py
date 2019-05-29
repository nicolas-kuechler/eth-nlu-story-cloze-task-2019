import sys
sys.path.append('../')
import pandas as pd 
import numpy as np

#Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
#nltk.download('punkt')

from global_variables import PATH_TRAIN

data = pd.read_csv(PATH_TRAIN, index_col=False).loc[:,'sentence1':'sentence5']
data = np.array(data.values.tolist())

f = lambda x: " ".join(x)
data = np.array(list(map(f, data)))

print(f'first 5: \n{data[:5]}')

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]

max_epochs = 100
vec_size = 20
alpha = 0.025

model = Doc2Vec(vector_size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
  
model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved")
