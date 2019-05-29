import sys
sys.path.append('../')
import pandas as pd 
import numpy as np
import csv

#Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
#nltk.download('punkt')

#from global_variables import PATH_TRAIN

def main():

    PATH_TRAIN = "./data/train_stories.csv"

    ## load data
    model = Doc2Vec.load('./code/dataset_generation/title_similarity.model')
    training_data = pd.read_csv(PATH_TRAIN, index_col=False).loc[:,'storytitle']

    print(f'length of training data: {len(training_data)}')

    ## go over all stories and find top 20   
    result = []
    for i in range(len(training_data)):
        tag = str(i)
        tfh = model.docvecs.most_similar(tag, topn=20)
        tfh_tags = [[tag,i[0]] for i in tfh]
        result += tfh_tags

        if i % 1000 == 0:
            print(f'story {i}')


    print(f'length of result: {len(result)}')
    
    res = np.array(result)
    print(f'res shape: {res.shape}')

    pd.DataFrame(res).to_csv('top_20_titles.csv', header=None, index=None)

if __name__ == '__main__':
    main()