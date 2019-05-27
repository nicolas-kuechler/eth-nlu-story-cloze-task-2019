import tokenization

import csv
import pandas as pd
from tqdm import tqdm



def get_train_dataset_true(path_train, shuffle=False):
    """
    
    """

    df = pd.read_csv(path_train, index_col=False)

    ds = []

    print("Processing Training Dataset...")
    for idx, storyid, storytitle, s1, s2, s3, s4, s5 in tqdm(df.itertuples(), total=df.shape[0]):
        story_start = ' '.join([s1, s2, s3, s4])
        
        # generate true sample
        true_story_end = s5
        positive_sample = {"story_start_id":storyid, "story_end_id":storyid, "story_start":story_start, "story_end": true_story_end, "label":1}
        ds.append(positive_sample)
       

    return ds

VOCAB_FILE = "./data/uncased_L-12_H-768_A-12/vocab.txt"

tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=True)
ds_train = get_train_dataset_true(path_train='./data/train_stories.csv')

token_sizes = []

for sample in tqdm(ds_train):
    start = tokenizer.tokenize(sample['story_start'])
    end = tokenizer.tokenize(sample['story_end'])
    print("Start:")
    print(start)
    print("End:")
    print(end)

    token_sizes.append((len(start), len(end)))

with open('./token_sizes.csv','w', newline='') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['start','end'])
    for row in token_sizes:
        csv_out.writerow(row)

