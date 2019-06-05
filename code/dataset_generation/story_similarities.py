'''
Needs to be run from within code/dataset_generation
Trains an embedding on the entire stories in which all the names were replaces by a tag. 
'''

import sys
sys.path.append('../')
import pandas as pd 
import numpy as np

#Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
#nltk.download('punkt')

PATH_TRAIN = '../../data/train_stories.csv'

data = pd.read_csv(PATH_TRAIN, index_col=False).loc[:,'sentence1':'sentence5']
data = np.array(data.values.tolist())

f = lambda x: " ".join(x)
data = np.array(list(map(f, data)))

def _remove_person(text, names):
    for word in text.split(' '):
        if word in names:
            text = text.replace(word, "<unk>")
    return text
    
names = pd.read_csv("../../data/first_names.csv", index_col=False)
names = set(names['Name'])

for i in range(len(data)):
    data[i] = _remove_person(text=data[i], names=names)
    #print(f'new title: {story_titles[i]}')
    if i % 1000 == 0:
        print(f'title {i}')

print(f'story titles length: {len(data)}')

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

model.save("story_similarity.model")
print("Model Saved")
