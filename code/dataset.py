import pandas as pd
import random
from tqdm import tqdm
import numpy as np



"""
  _______        _       _             
 |__   __|      (_)     (_)            
    | |_ __ __ _ _ _ __  _ _ __   __ _ 
    | | '__/ _` | | '_ \| | '_ \ / _` |
    | | | | (_| | | | | | | | | | (_| |
    |_|_|  \__,_|_|_| |_|_|_| |_|\__, |
                                  __/ |
                                 |___/ 
"""

def get_train_dataset(path_train, path_names=None, shuffle=True):
    """
    get the training dataset with randomly sampled endings and possible name replacement
    """
    
    df = pd.read_csv(path_train, index_col=False)
    ds = []

    # load name replacement dict
    if path_names:
        names = pd.read_csv(path_names, index_col=False)
        names = set(names['Name'])

    np.random.seed(1234)

    print("Processing Training Dataset...")
    for idx, storyid, storytitle, s1, s2, s3, s4, s5 in tqdm(df.itertuples(), total=df.shape[0]):
       
        story_start = ' '.join([s1, s2, s3, s4])
        true_story_end = s5
        
        # generate false sample by using random ending
        while True:
            false_ind = np.random.randint(0, df.shape[0])
            if false_ind != idx:
                break

        false_story_end = df.loc[false_ind, 'sentence5']
        false_story_id = df.loc[false_ind, 'storyid']

        # do name replacement
        if path_names:
            false_story_end = _adjust_false_story_end(story_start=story_start, true_story_end=true_story_end, false_story_end=false_story_end, names=names)

        negative_sample = {"story_start_id":storyid, "story_end_id":false_story_id, "story_start":story_start, "story_end": false_story_end, "label":0}
        ds.append(negative_sample)

        # generate true sample
        positive_sample = {"story_start_id":storyid, "story_end_id":storyid, "story_start":story_start, "story_end": true_story_end, "label":1}
        ds.append(positive_sample)
       
    if shuffle:
        random.shuffle(ds)

    return ds

def get_train_dataset_true(path_train, shuffle=False):
    """
    generate the training dataset containing only the true samples (label=1)
    """

    df = pd.read_csv(path_train, index_col=False)

    ds = []

    print("Processing Training Dataset...")
    for idx, storyid, storytitle, s1, s2, s3, s4, s5 in df.itertuples():
        story_start = ' '.join([s1, s2, s3, s4])
        
        # generate true sample
        true_story_end = s5
        positive_sample = {"story_start_id":storyid, "story_end_id":storyid, "story_start":story_start, "story_end": true_story_end, "label":1}
        ds.append(positive_sample)
       
    if shuffle:
        random.shuffle(ds)

    return ds

def get_crossproduct_dataset_false(path_train, path_mapping, path_names):
    """
    generate a cross product dataset with all endings for a sample frm the file path_train defined in the file path_mapping
    """

    df = pd.read_csv(path_train, index_col=False)

    df_mapping = pd.read_csv(path_mapping, index_col=False, names=['start_idx', 'end_idx'])

    names = pd.read_csv(path_names, index_col=False)
    names = set(names['Name'])

    ds = []

    print("Processing Cross Product Train Dataset...")
    for idx, storyid, storytitle, s1, s2, s3, s4, s5 in tqdm(df.itertuples(), total=df.shape[0]):
        story_start = ' '.join([s1, s2, s3, s4])
        
        # look up which are the most similar 
        sample_mappings = df_mapping.loc[df_mapping['start_idx'] == idx]
        true_story_end = s5

        false_endings = df.loc[sample_mappings['end_idx']][['storyid', 'sentence5']]
       
        # generate all false samples
        for _, false_storyid, false_story_end in tqdm(false_endings.itertuples(), total=false_endings.shape[0]):
            false_story_end = _adjust_false_story_end(story_start=story_start, true_story_end=true_story_end, false_story_end=false_story_end, names=names)
            negative_sample = {"story_start_id":storyid, "story_end_id":false_storyid, "story_start":story_start, "story_end": false_story_end, "label":0}
            ds.append(negative_sample)
        
    return ds

def get_ablation1_dataset(path_train='./data/train_stories.csv', shuffle=True):
    """
    dataset using random false endings
    """
    return get_train_dataset(path_train=path_train, shuffle=shuffle)

def get_ablation2_dataset(path_train='./data/train_stories.csv', path_names='./data/first_names.csv', shuffle=True):
    """
    dataset using random false endings
    + name replacement
    """
    return get_train_dataset(path_train=path_train, path_names=path_names, shuffle=shuffle)

def get_ablation3_dataset(path_train='./data/train_stories.csv', path_mapping='./data/train_stories_top_1_most_similar_titles.csv', path_names='./data/first_names.csv', shuffle=True):
    """
    dataset using top 1 most similar title ending
    + name replacement
    """

    ds_false = get_crossproduct_dataset_false(path_train, path_mapping, path_names)
    ds_true = get_train_dataset_true(path_train, shuffle=False)

    assert(len(ds_false)==len(ds_true))

    ds = ds_false + ds_true

    if shuffle:
        random.shuffle(ds)

    return ds

def get_ablation4_dataset(path_train='./data/train_stories.csv', path_mapping='./data/train_stories_top_1_most_similar_stories.csv', path_names='./data/first_names.csv', shuffle=True):
    """
    dataset using top 1 most similar story ending
    + name replacement
    """
    return get_ablation3_dataset(path_train, path_mapping, path_names, shuffle)
    

"""
 __      __   _ _     _       _   _                        _______        _   
 \ \    / /  | (_)   | |     | | (_)               ___    |__   __|      | |  
  \ \  / /_ _| |_  __| | __ _| |_ _  ___  _ __    ( _ )      | | ___  ___| |_ 
   \ \/ / _` | | |/ _` |/ _` | __| |/ _ \| '_ \   / _ \/\    | |/ _ \/ __| __|
    \  / (_| | | | (_| | (_| | |_| | (_) | | | | | (_>  <    | |  __/\__ \ |_ 
     \/ \__,_|_|_|\__,_|\__,_|\__|_|\___/|_| |_|  \___/\/    |_|\___||___/\__|
                                                                              
"""

def get_valid_dataset(path_valid, shuffle=False):
    """
    Standard Validation Dataset for Story Cloze Task
    """
    df = pd.read_csv(path_valid, index_col=False)

    ds = []

    print("Processing Valid Dataset...")
    for idx, storyid, s1, s2, s3, s4, ending1, ending2, right_end in tqdm(df.itertuples(), total=df.shape[0]):
        story_start = ' '.join([s1, s2, s3, s4])

        if right_end == 1:
            true_story_end = ending1
            false_story_end = ending2
        elif right_end == 2:
            true_story_end = ending2
            false_story_end = ending1
        else:
            raise ValueError("Not correct end!")

        positive_sample = {"story_start_id":storyid + "_1", "story_end_id":storyid + "_1", "story_start":story_start, "story_end": true_story_end, "label":1}
        ds.append(positive_sample)

        negative_sample = {"story_start_id":storyid + "_1", "story_end_id":storyid + "_0", "story_start":story_start, "story_end": false_story_end, "label":0}
        ds.append(negative_sample)

    if shuffle:
        random.shuffle(ds)

    return ds

def get_test_dataset(path_test, shuffle=False):
    """
    Standard Test Dataset for Story Cloze Task
    """
    return get_valid_dataset(path_valid=path_test, shuffle=shuffle)

def get_eth_test_dataset(path_test):
    """
    Test Dataset provided by ETH (without label)
    """
    df = pd.read_csv(path_test, index_col=False)

    ds = []

    print("Processing ETH TEST Dataset...")
    for idx, s1, s2, s3, s4, ending1, ending2 in tqdm(df.itertuples(), total=df.shape[0]):
        story_start = ' '.join([s1, s2, s3, s4])

        story_id = str(idx)
        # label all as zero because BERT requires valid label
        sample1 = {"story_start_id": story_id, "story_end_id": story_id + "_1", "story_start":story_start, "story_end": ending1, "label":0}
        ds.append(sample1)

        sample2 = {"story_start_id": story_id, "story_end_id": story_id + "_2", "story_start":story_start, "story_end": ending2, "label":0}
        ds.append(sample2)

    return ds

def generate_and_save_init_test_results(path_cross, path_test_results):
    """
    Generates and stores a file with prob0 and prob1 = 0.5 
    """

    df = pd.read_csv(path_cross, sep='\t', index_col=False)

    n_samples = df.shape[0]

    df = pd.DataFrame(0.5, index=range(n_samples), columns=['prob0', 'prob1'])

    df.to_csv(path_test_results, sep='\t', index=False, header=False)

"""
  _____                            _    _                 _     _   _      
 |  __ \                          | |  | |               (_)   | | (_)     
 | |__) |__ _ __ ___  ___  _ __   | |__| | ___ _   _ _ __ _ ___| |_ _  ___ 
 |  ___/ _ \ '__/ __|/ _ \| '_ \  |  __  |/ _ \ | | | '__| / __| __| |/ __|
 | |  |  __/ |  \__ \ (_) | | | | | |  | |  __/ |_| | |  | \__ \ |_| | (__ 
 |_|   \___|_|  |___/\___/|_| |_| |_|  |_|\___|\__,_|_|  |_|___/\__|_|\___|
                                                                           
"""

def _adjust_false_story_end(story_start, true_story_end, false_story_end, names):
    person_story_start = _extract_person(story_start, names)

    # extract person from sampled ending and replace with person of story start
    if person_story_start is not None:
        person_false_story_end = _extract_person(false_story_end, names)
        if person_false_story_end is not None:
            false_story_end = _replace_person(text=false_story_end, old_person=person_false_story_end, new_person=person_story_start)
    
    return false_story_end


def _replace_person(text, old_person, new_person):
    if old_person == 'He' or old_person == 'She' or old_person == 'They' or old_person== 'I':
        return text.replace(old_person + " ", new_person + " ")
    if old_person == 'he' or old_person == 'she' or old_person == 'they':
        return text.replace(" " + old_person + " ", " " + new_person + " ")

    return text.replace(old_person, new_person)

def _extract_person(text, names):

    persons = set()
    
    for word in text.split(' '):
        if word in names:
            persons.add(word)

    if len(persons) >= 1:
        for p in persons:
            return p
    else:
        if 'He ' in text:
            return 'He'
        if ' he ' in text:
            return 'he'
        if 'She ' in text:
            return 'She'
        if ' she ' in text:
            return 'she'
        if 'They ' in text:
            return 'They'
        if ' they ' in text:
            return 'they'
        if 'I ' in text:
            return 'I'

def save_dataset_as_tsv(dataset, path):
    df = pd.DataFrame(dataset)
    df.to_csv(path, sep='\t', index=False, columns=['story_start_id', 'story_end_id', 'story_start', 'story_end', 'label'])

def main():

    ds_train_true = get_train_dataset_true(path_train='./data/train_stories.csv', shuffle=False)
    save_dataset_as_tsv(ds_train_true, path="./data/ds_train_true.tsv")

    """
    # These are commented out because the ablation.py already generates them when needed

    ds_train_ablation1 = get_ablation1_dataset()
    save_dataset_as_tsv(ds_train_ablation1, path="./data/ds_train_ablation1.tsv")

    ds_train_ablation2 = get_ablation2_dataset()
    save_dataset_as_tsv(ds_train_ablation2, path="./data/ds_train_ablation2.tsv")
    """
    ds_train_ablation3 = get_ablation3_dataset()
    save_dataset_as_tsv(ds_train_ablation3, path="./data/ds_train_ablation3.tsv")
    """
    ds_train_ablation4 = get_ablation4_dataset()
    save_dataset_as_tsv(ds_train_ablation4, path="./data/ds_train_ablation4.tsv")
    """

    ds_valid = get_valid_dataset(path_valid='./data/cloze_test_val__spring2016 - cloze_test_ALL_val.csv', shuffle=False)
    save_dataset_as_tsv(ds_valid, path="./data/ds_valid.tsv")

    ds_test = get_test_dataset(path_test='./data/test_for_report-stories_labels.csv', shuffle=False)
    save_dataset_as_tsv(ds_test, path="./data/ds_test.tsv")

    ds_eth_test = get_eth_test_dataset(path_test='./data/test-stories.csv')
    save_dataset_as_tsv(ds_eth_test, path="./data/ds_eth_test.tsv")

    ds_cross_product_false = get_crossproduct_dataset_false(path_train='./data/train_stories.csv', path_mapping='./data/train_stories_top_20_most_similar_titles.csv', path_names='./data/first_names.csv')
    save_dataset_as_tsv(ds_cross_product_false, path="./data/ds_cross_product_false.tsv")

    generate_and_save_init_test_results(path_cross="./data/ds_cross_product_false.tsv",  path_test_results="./data/init/cross_results.tsv")

if __name__ == '__main__':
    main()