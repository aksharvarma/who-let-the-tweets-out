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

def avg_words_per_sentence(text):
    sentences = text.split('.')
    return np.mean([len(sentence.split()) for sentence in sentences])


def avg_chars_per_word(text):
    words = text.split()
    return sum(len(word) for word in words)/len(words)


def num_of_punctutations(text):
    def is_punctutation(c):
        return ((not c.isalpha()) and
                (not c.isdigit()) and
                (not c.isspace()))

    return len([c for c in text if is_punctutation(c)])


def num_of_chars(text):
    return len(text)


def num_of_words(text):
    return len([w for w in text.split() if w != ''])


def extract_features(tweets, keep_orig=False):
    embedding = pd.DataFrame(tweets["embedding"].values.tolist())

    if keep_orig:
        cols_to_keep = tweets.columns
    else:
        cols_to_keep = ['avg_chars_per_word', 'avg_words_per_sentence',
                        'num_of_chars', 'num_of_punctutations', 'user_id']

    return pd.concat([tweets[cols_to_keep], embedding])
