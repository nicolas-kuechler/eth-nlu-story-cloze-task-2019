import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix, vstack, hstack
from scipy import sparse

def auc(y_true,y_score):
    y_true=tf.keras.utils.to_categorical(y_true,num_classes=4)
    y_score=tf.keras.utils.to_categorical(y_score,num_classes=4)
    return roc_auc_score(y_true=y_true,y_score=y_score, average='macro')

def f1(y_true,y_score):
    #y_true=tf.keras.utils.to_categorical(y_true,num_classes=4)
    #y_score=tf.keras.utils.to_categorical(y_score,num_classes=4)
    return f1_score(y_true=y_true,y_pred=y_score,average='macro')

def calculate_and_print_scores(y_train_true,y_train_pred, y_val_true, y_val_pred):
    #calculate actual score

    confusion = tf.math.confusion_matrix(y_val_true,y_val_pred,num_classes=4).numpy()

    print(confusion)

    auc_roc_train = 0.0#auc(y_true=y_train_true,y_score=y_train_pred)
    auc_roc_val = 0.0#auc(y_true=y_val_true,y_score=y_val_pred)
    f1_train = f1(y_true=y_train_true,y_score=y_train_pred)
    f1_val = f1(y_true=y_val_true,y_score=y_val_pred)

    print(f"auc_roc_train: {str(auc_roc_train)} - "+ \
        f"auc_roc_val: {str(auc_roc_val)} - "+ \
        f"f1_train: {str(f1_train)} - "+ \
        f"f1_val: {str(f1_val)}")
    return

def preprocess_body_body(body_body_train,body_body_val,lim_unigram):
    #load data and rename column
    data_train  = pd.read_csv(body_body_train,sep='\t',usecols=['story_start','story_end','label'])
    data_val  = pd.read_csv(body_body_val,sep='\t',usecols=['story_start','story_end','label'])
    
    data_train.columns = ['body1','body2','stance']
    data_val.columns = ['body1','body2','stance']

    #get unique bodys from train data
    train1 = ((data_train.copy())[['body1']]).drop_duplicates().values.reshape(-1)
    val1 = data_val.copy()[['body1']].drop_duplicates().values.reshape(-1)
    train2 = data_train.copy()[['body2']].drop_duplicates().values.reshape(-1)
    val2 = data_val.copy()[['body2']].drop_duplicates().values.reshape(-1)

    
    train = np.concatenate([train1,train2])
    val = np.concatenate([val1,val2])

    del train1
    del train2
    del val1
    del val2

    write_log('start tfidf')
    # Create vectorizers and BOW and TF arrays for train set
    tfreq = TfidfVectorizer(max_features=lim_unigram, stop_words=stop_words,use_idf=False).\
        fit(train)  # Train set only
    
    write_log('start idf')
    tfidf_vectorizer = TfidfVectorizer(max_features=lim_unigram, stop_words=stop_words).\
        fit(np.concatenate((train,val),axis = 0))  # Train and test sets

    write_log('end idf')

    del train
    del val

    write_log('start train generation')
    train_set = data_train[['body1','body2']].values
    train_set = mapper(train_set,tfreq,tfidf_vectorizer)
    
    write_log('start test generation')
    test_set = data_val[['body1','body2']].values
    test_set = mapper(test_set,tfreq,tfidf_vectorizer)

    write_log('end test generation')

    #getting labels and directly mapping them to numbers
    train_stances = np.vectorize(lambda label: int(label))(data_train[['stance']].values)
    val_stance = np.vectorize(lambda label: int(label))(data_val[['stance']].values)

    n_train = train_stances.shape[0]
    feature_size = 2*lim_unigram+1
    return train_set, train_stances,test_set, n_train, feature_size, val_stance

def mapper(train_set, tfreq,tfidf_vectorizer):

    #data = []
    data_sparse = []
    for i in range(train_set.shape[0]):
        head_tf = tfreq.transform([train_set[i,0]])
        body_tf = tfreq.transform([train_set[i,1]])

        head_tfidf = tfidf_vectorizer.transform([train_set[i,0]])
        body_tfidf = tfidf_vectorizer.transform([train_set[i,1]])

        tfidf_cos = cosine_similarity(head_tfidf.toarray(), body_tfidf.toarray())[0].reshape(1, 1)

        #entry = np.squeeze(np.c_[head_tf.toarray(), body_tf.toarray(), tfidf_cos])
        #data.append(entry)

        entry = hstack([head_tf,body_tf,tfidf_cos])
        data_sparse.append(entry)
    
    return vstack(data_sparse)

def save_new_body_body(file_body_body_train, file_body_body_val,save_dir, lim_unigram):
    train_set, train_stances,test_set, n_train, feature_size, val_stance = \
        preprocess_body_body(file_body_body_train, file_body_body_val,lim_unigram = lim_unigram)

    save_all_data(save_dir,train_set, train_stances,test_set, n_train, feature_size)
    save_np(save_dir+'val_stance',val_stance)
    return

def load_body_body(save_dir):
    train_set, train_stances,test_set, n_train, feature_size = load_all_data(save_dir)
    return train_set, train_stances,test_set, n_train, feature_size , load_np(save_dir+'val_stance')

def save_np(path, np_array, verbose = 1):
    path=path+'.npy'
    np.save(path,np_array)

    if verbose == 1:
        print(f"saved: {path} with shape: {np_array.shape}")

    return

def load_np(path, verbose = 1):
    path=path+'.npy'
    load = np.load(path)

    if verbose == 1:
        print(f"loaded: {path} with shape: {load.shape}")

    return load

def save_npz(path, np_array, verbose = 1):
    path=path+'.npz'
    sparse.save_npz(path,np_array)

    if verbose == 1:
        print(f"saved: {path} with shape: {np_array.shape}")

    return

def load_npz(path, verbose = 1):
    path=path+'.npz'
    load = sparse.load_npz(path)

    if verbose == 1:
        print(f"loaded: {path} with shape: {load.shape}")

    return load

def save_all_data(save_dir,train_set, train_stances,test_set, n_train, feature_size):
    #save to file
    #save_np(save_dir+'train_set',train_set)
    save_npz(save_dir+'train_set',train_set)
    save_np(save_dir+'train_stance',train_stances)
    #save_np(save_dir+'test_set',test_set)
    save_npz(save_dir+'test_set',test_set)

    save_np(save_dir+'variables',np.array([n_train,feature_size]))

    return

def load_all_data(save_dir):

    #train_set = load_np(save_dir+'train_set')
    train_set = load_npz(save_dir+'train_set')
    train_stances = load_np(save_dir+'train_stance')

    #test_set = load_np(save_dir+'test_set')
    test_set = load_npz(save_dir+'test_set')

    variables = load_np(save_dir+'variables')

    return train_set.tocsr(), train_stances, test_set.tocsr(), variables[0], variables[1]

def save_predictions(pred, file):
    out = open(file, 'w')

    for i,elem in enumerate(pred):
        array=[str(elem[0]), str(elem[1])]
        out.write('\t'.join(array) + "\n")
    out.close()

    return 

def batch_generator(X_data, y_data, batch_size):
    #from https://stackoverflow.com/questions/41538692/using-sparse-matrices-with-keras-and-tensorflow
    #accessed 25.05
    samples_per_epoch = X_data.shape[0]
    number_of_batches = samples_per_epoch//batch_size
    index = np.arange(np.shape(y_data)[0])
    for counter in range(int(number_of_batches)):
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X_data[index_batch,:].todense()
        y_batch = y_data[index_batch]
        yield (np.array(X_batch),y_batch)

stop_words = [
        "a", "about", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along",
        "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
        "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be",
        "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
        "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "co",
        "con", "could", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight",
        "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
        "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill", "find", "fire", "first", "five", "for",
        "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had",
        "has", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself",
        "him", "himself", "his", "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed", "interest",
        "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made",
        "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much",
        "must", "my", "myself", "name", "namely", "neither", "nevertheless", "next", "nine", "nobody", "now", "nowhere",
        "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours",
        "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see",
        "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some",
        "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take",
        "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby",
        "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though",
        "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve",
        "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what",
        "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon",
        "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will",
        "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"
        ]

def write_log(text):
    with open("out.txt", "a") as myfile:
        myfile.write(text+'\n')
    print(text)
    return 

if __name__ == '__main__':
    pass
