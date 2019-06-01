import os
import pandas as pd
import numpy as np
import math
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

def baseline(X,Y):
    '''
    This file computes a baseline 
    '''

    length = math.floor(X.shape[0]*0.9)

    print(f'length: {length}')
    X_train = X[:length]
    Y_train = Y[:length]

    test_labels = Y[length:]
    test_data = X[length:]

    print(X_train.shape,test_data.shape)
    #MultiLayerPerceptronClassifier
    mlp = MLPClassifier()
    mlp = mlp.fit(X_train, Y_train)
    mlp_prediction = mlp.predict(test_data)
    #print(mlp_prediction)

    #DecisionTreeClassifier
    dtc_clf = tree.DecisionTreeClassifier()
    dtc_clf = dtc_clf.fit(X_train,Y_train)
    dtc_prediction = dtc_clf.predict(test_data)
    #print(dtc_prediction)

    #RandomForestClassifier
    rfc_clf = RandomForestClassifier()
    rfc_clf.fit(X_train,Y_train)
    rfc_prediction = rfc_clf.predict(test_data)
    #print(rfc_prediction)

    #LogisticRegression
    l_clf = LogisticRegression()
    l_clf.fit(X_train,Y_train)
    l_prediction = l_clf.predict(test_data)
    #print(l_prediction)

    #Support Vector Classifier
    s_clf = SVC()
    s_clf.fit(X_train,Y_train)
    s_prediction = s_clf.predict(test_data)
    #print(s_prediction)


    #accuracy scores
    mlp_acc = accuracy_score(mlp_prediction, test_labels)
    dtc_tree_acc = accuracy_score(dtc_prediction,test_labels)
    rfc_acc = accuracy_score(rfc_prediction,test_labels)
    l_acc = accuracy_score(l_prediction,test_labels)
    s_acc = accuracy_score(s_prediction,test_labels)

    classifiers = ['MLP', 'Decision Tree', 'Random Forest', 'Logistic Regression' , 'SVC']
    accuracy = np.array([mlp_acc, dtc_tree_acc, rfc_acc, l_acc, s_acc])
    max_acc = np.argmax(accuracy)
    print(classifiers[max_acc] + ' is the best classifier for this problem')
    print('accuracies: ', accuracy)
    print(f'max accuracy: {accuracy[max_acc]}')

    return

if __name__ == '__main__':
    if 'SCRATCH' in os.environ:
        path_scratch = os.environ['SCRATCH']
    else: 
        path_scratch = '../../../../scratch/brunnelu/sct_2/'
    


    file_result = path_scratch+'test_results.tsv'

    path_val = '../data/sct/val_1.tsv'

    df_res = pd.read_csv(file_result, sep='\t', header=None)
    df_gold = pd.read_csv(path_val, sep='\t', header=0)
    
    gold_label = np.array(df_gold['label'])
    res = np.array(df_res)

    print(gold_label.shape)
    print(res.shape)

    length = res.shape[0]//2

    gold_label = gold_label.reshape((length,-1))
    res = res.reshape((length,-1))

    print(gold_label.shape)
    print(res.shape)

    X = []
    Y = []

    for i,elem in enumerate(res):
        X.append([elem[0],elem[1],elem[2],elem[3]])
        Y.append(gold_label[i,0])

    X= np.array(X)
    Y= np.array(Y).reshape((-1,1))

    print(X,Y)
    print(X.shape,Y.shape)

    X,Y =shuffle(X,Y)
    baseline(X,Y)