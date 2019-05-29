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

from global_variables import PATH_TRAIN

def main():

    ## load data
    model = Doc2Vec.load('title_similarity.model')
    training_data = pd.read_csv(PATH_TRAIN, index_col=False).loc[:,'storytitle']
    training_data = np.array(training_data.values.tolist())

    print(f'length of training data: {len(training_data)}')

    ## go over all stories and find top 500   
    for i in range(10):
        tag = str(i)
        tfh = model.docvecs.most_similar(tag, topn=5)
        tfh_tags = [[tag,j[0]] for j in tfh]
        print(f'reference story: \n{training_data[i]}')
        for _, tag in tfh_tags:
            print(f'similar story: \n{training_data[int(tag)]}')

        print('\n\n')

       


  

if __name__ == '__main__':
    main()