'''
Needs to be run from within code/dataset_generation
Trains an embedding on the story titles in which the names have been replaced by tags.
'''

import sys
sys.path.append('../')
import pandas as pd 
import numpy as np

#Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

from global_variables import PATH_TRAIN

story_titles = pd.read_csv(PATH_TRAIN, index_col=False).loc[:,'storytitle']
story_titles = np.array(story_titles.values.tolist())

#story_titles = story_titles[:20]

def _remove_person(text, names):
    for word in text.split(' '):
        if word in names:
            text = text.replace(word, "<unk>")
    return text
    
names = pd.read_csv("../../data/first_names.csv", index_col=False)
names = set(names['Name'])

for i in range(len(story_titles)):
    story_titles[i] = _remove_person(text=story_titles[i], names=names)
    #print(f'new title: {story_titles[i]}')
    if i % 1000 == 0:
        print(f'title {i}')

print(f'story titles: {story_titles}')

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(story_titles)]

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

model.save("title_similarity.model")
print("Model Saved")
