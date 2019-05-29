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

def similarity_list():
    ## load model
    model= Doc2Vec.load("d2v.model")

    ## used for similarity computations
    training_data = pd.read_csv(PATH_TRAIN, index_col=False).loc[:,'sentence1':'sentence5']
    data = np.array(training_data.values.tolist())
    f = lambda x: " ".join(x)
    data_concat = np.array(list(map(f, data)))

    ## load first 4
    first_4 = training_data.loc[:, 'sentence1':'sentence4']
    first_4 = np.array(first_4.values.tolist())

    ## load endings
    endings = training_data.loc[:, 'sentence5']
    endings = np.array(endings.values.tolist())
    f = lambda x: " ".join(x)
    first_4 = np.array(list(map(f, first_4)))
    print(f'shape of first 4 before repeating and after joining: {first_4.shape}')

    ## correct block
    index_col = np.arange(len(data_concat))
    data_ind_j_lab = np.column_stack((index_col, first_4, endings, np.ones(len(data_concat)).astype(int)))
    print(f'shape of data after indexing and labelling: {data_ind_j_lab.shape}')
    print(f'first 3: \n{data_ind_j_lab[:3]}')

    ind = len(data_ind_j_lab)
    tuplify = lambda x: tuple(x)
    res_list = list(map(tuplify, data_ind_j_lab.tolist()))
    print(f'tuple list first 3: \n{res_list[:3]}')

    ## find the top 1 similar sentence
    for i in range(len(data_ind_j_lab)):
        tag = str(i)
        ms = model.docvecs.most_similar(tag, topn=1)
        #print(f'most similar: {type(ms[0][0])}')
        ms_tag = int(ms[0][0])
        tup = (str(ind), first_4[i], endings[ms_tag], '0')
        res_list.append(tup)
        ind += 1
        #print(f'appended tuple: \n{tup}')
        if i % 1000 == 0:
            print(f'appending tuple: {i}')
        
    print(f'length of new result list: {len(res_list)}')
    return res_list

if __name__ == '__main__':
    similarity_list()