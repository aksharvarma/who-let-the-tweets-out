import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import nltk
from sklearn.model_selection import StratifiedKFold

def read_data_from_file(filename):
    print('Reading tweets from:', filename)
    return pd.read_pickle(filename)

def get_X_y(tweets):
    x_cols = [col for col in tweets.columns if col != 'user_id']
    X = np.array(tweets.loc[:, x_cols], dtype=np.float64)
    np.nan_to_num(X, copy=False)  # Fix nan issues in in_reply_to_user_id
    y = np.array(tweets.user_id.values, dtype=np.int32)
    return X, y, x_cols

def get_test_train_split_generator(X, y, k=10):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=55861)
    return skf.split(X, y)

def extract_features(tweets, keep_orig=False):
    def avg_words_per_sentence(tweet):
        # tokenizer = nltk.tokenize.RegexpTokenizer('\.')
        # sentences = tokenizer.tokenize(tweet)
        sentences = tweet.split('.')
        return np.mean([len(sentence.split()) for sentence in sentences])

    def avg_chars_per_word(tweet):
        # tokenizer = nltk.tokenize.RegexpTokenizer('\w+|\S+')
        words = tweet.split()
        # if len(words) == 0:
        #     print(tweet, flush=True)
        #     return sum(len(word) for word in words)/(len(words)+1)
        return sum(len(word) for word in words)/len(words)

    def num_of_punctutations(tweet):
        def is_punctutation(c):
            return ((not c.isalpha()) and
                    (not c.isdigit()) and
                    (not c.isspace()))

        return len([c for c in tweet if is_punctutation(c)])

    def num_of_chars(tweet):
        return len(tweet)

    def num_of_words(tweet):
        return len([w for w in tweet.split() if w != ''])

    # def pos_via_nltk(tweet):
    #     return 0

    print('Extracting text features...', end='', flush=True)
    tweets['avg_chars_per_word'] = tweets.text_body.map(avg_chars_per_word)
    tweets['avg_words_per_sentence'] = tweets.text_body.map(avg_words_per_sentence)
    tweets['num_of_chars'] = tweets.text_body.map(num_of_chars)
    tweets['num_of_punctutations'] = tweets.text_body.map(num_of_punctutations)
    tweets = tweets.drop(['text_body'], axis=1)
    # tweets['pos_via_nltk'] = tweets.text_body.map(pos_via_nltk)
    print('Done.')

    if keep_orig:
        cols_to_keep = tweets.columns
    else:
        cols_to_keep = ['avg_chars_per_word', 'avg_words_per_sentence',
                    'num_of_chars', 'num_of_punctutations', 'user_id']

    return tweets[cols_to_keep]
