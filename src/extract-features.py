import numpy as np
import pandas as pd
import progressbar
from collections import Counter
from functools import reduce
from sklearn.decomposition import TruncatedSVD
import pickle
import utils
import argparse


def compute_ngram_embedding(tweet_text_body_tokens, n, dim):
    X_ngram = np.zeros((len(tweet_text_body_tokens), dim))
    row_index = 0

    def update_ngram_embedding(row_index):
        tokens = tweet_text_body_tokens[row_index]
        l = len(tokens)
        for i in range(l - (n - 1)):
            col_index = hash("".join(tokens[i:i+n])) % dim
            X_ngram[row_index, col_index] += 1

    row_count = len(tweet_text_body_tokens)
    update_ngram_embedding_with_progress = with_progress(
        update_ngram_embedding,
        row_count,
        prefix = "{n}-gram ".format(n = n))

    for row_index in range(row_count):
        update_ngram_embedding_with_progress(row_index)

    return X_ngram


def compute_linguistic_features(tweet_text_body):
    n = len(tweet_text_body)
    mapper = tweet_text_body.map
    return np.array([
        mapper(with_progress(utils.avg_chars_per_word, n, prefix = "Avg. Chars ")),
        mapper(with_progress(utils.avg_words_per_sentence, n, prefix = "Avg. Words ")),
        mapper(with_progress(utils.num_of_chars, n, prefix = "Num. Chars ")),
        mapper(with_progress(utils.num_of_punctutations, n, prefix = "Num. Puncts "))
    ]).transpose()


def with_progress(fun, max_value, increment = 5000, prefix = ""):
    bar = progressbar.ProgressBar(max_value = max_value, prefix = prefix)
    i = 0

    def apply_and_update(*args, **kwargs):
        nonlocal i
        res = fun(*args, **kwargs)
        i = i + 1
        if i % increment == 0:
            bar.update(i)
        elif i == max_value:
            bar.finish()
        return res

    return apply_and_update


def tokenize_strings(tweet_text_body):
    def tokenize(text):
        words = []
        word = ""
        for c in text:
            if c.isalpha():
                word += c
            else:
                words.append(word)
                word = ""
                if not c.isspace():
                    words.append(c)
                    words.append(word)
        return list(filter(lambda word: len(word) > 0, words))

    n = len(tweet_text_body)

    return list(map(with_progress(tokenize, n, prefix = "Tokenizing "),
                    tweet_text_body))


def compute_word_weights(tweet_text_body_tokens, a = 10e-3):
    def update_counter(counter, values):
        counter.update(values)
        return counter
    n = len(tweet_text_body_tokens)
    word_weight_mapping = reduce(with_progress(update_counter, n, prefix = "Weighing "),
                                 tweet_text_body_tokens,
                                 Counter())
    unique_word_count = len(word_weight_mapping.keys())
    for key, value in word_weight_mapping.items():
        frequency = value / unique_word_count
        word_weight_mapping[key] = a / (a + frequency)
    return word_weight_mapping

def index(words,
          word_index_mapping,
          word_weight_mapping):
    word_indices = []
    word_weights = []
    for word in words:
        index = word_index_mapping.get(word, -1)
        weight = word_weight_mapping.get(word, -1)
        if index >= 0:
            word_indices.append(index)
            word_weights.append(weight)
    return (np.array(word_indices),
            np.array(word_weights))


def compute_sif_embedding_with_pc(tweet_text_body_tokens,
                                  word_weight_mapping,
                                  word_index_mapping,
                                  word_vector_mapping):

    dims = (len(tweet_text_body_tokens), word_vector_mapping.shape[1])
    sif_with_pc = np.zeros(dims)

    def embed(row_index):
        words = tweet_text_body_tokens[row_index]
        (word_indices, word_weights) = index(words,
                                             word_index_mapping,
                                             word_weight_mapping)
        if len(word_indices) == 0:
            embedding_with_pc = np.zeros(word_vector_mapping.shape[1])
        else:
            embedding_with_pc = np.sum(word_vector_mapping[word_indices, :] *
                                       word_weights[:, np.newaxis] /
                                       len(word_weights),
                                       axis = 0)
        sif_with_pc[row_index,:] = embedding_with_pc


    embed_with_progress = with_progress(embed, dims[0], prefix = "Embedding ")

    for row_index in range(len(tweet_text_body_tokens)):
        embed_with_progress(row_index)

    return sif_with_pc


def compute_sif_embedding_without_pc(embedding_with_pc):
    # compute the principal component
    svd = TruncatedSVD(n_components = 1, n_iter = 7, random_state = 0)
    svd.fit(embedding_with_pc)
    pc = svd.components_

    embedding_without_pc = (embedding_with_pc -
                            embedding_with_pc.dot(pc.transpose()) * pc)
    return embedding_without_pc


def parse_arguments():
    parser = argparse.ArgumentParser(description = ("Preprocess word vector " +
                                                    "file and store it in "
                                                    "native numpy format"))
    parser.add_argument('processed_tweets_filepath',
                        metavar = 'processed-tweets-filepath',
                        type = str,
                        help = "text file containing word embedding")
    parser.add_argument('processed_words_filepath',
                        metavar = 'processed-words-filepath',
                        type = str,
                        help = "pickle file containing word to index mapping")
    parser.add_argument('processed_vectors_filepath',
                        metavar = 'processed-vectors-filepath',
                        type = str,
                        help = "npz file containing embedding of words by their index")
    parser.add_argument('y_filepath',
                        metavar = 'processed-vectors-filepath',
                        type = str,
                        help = "npz file containing embedding of words by their index")
    parser.add_argument("x_bigram_filepath",
                        metavar = "x-bigram-filepath",
                        type = str,
                        help = "npy file containing bigram representation")
    parser.add_argument("x_trigram_filepath",
                        metavar = "x-trigram-filepath",
                        type = str,
                        help = "npy file containing trigram representation")
    parser.add_argument("x_sif_with_pc_filepath",
                        metavar = "x-sif-wih-pc-filepath",
                        type = str,
                        help = "npy file containing SIF with PC embedding")
    parser.add_argument("x_sif_without_pc_filepath",
                        metavar = "x-sif-wihout-pc-filepath",
                        type = str,
                        help = "npy file containing SIF without PC embedding")
    parser.add_argument("x_linguistic_feature_filepath",
                        metavar = "x-linguistic-feature-filepath",
                        type = str,
                        help = "npy file containing linguistic feature representation")

    return parser.parse_args()


def main():
    args = parse_arguments()

    tweets = pd.read_pickle(args.processed_tweets_filepath)

    np.save(args.y_filepath, tweets.user_id)

    tweet_text_body_tokens = tokenize_strings(tweets.text_body)

    word_weight_mapping = compute_word_weights(tweets.text_body)

    X_bigram = compute_ngram_embedding(tweet_text_body_tokens, 2, 300)
    np.save(args.x_bigram_filepath, X_bigram)

    X_trigram = compute_ngram_embedding(tweet_text_body_tokens, 3, 300)
    np.save(args.x_trigram_filepath, X_trigram)

    word_index_mapping = pickle.load(open(args.processed_words_filepath, "rb"))

    word_vector_mapping = np.load(args.processed_vectors_filepath)

    X_sif_with_pc = compute_sif_embedding_with_pc(tweet_text_body_tokens,
                                                  word_weight_mapping,
                                                  word_index_mapping,
                                                  word_vector_mapping)

    np.save(args.x_sif_with_pc_filepath, X_sif_with_pc)

    X_sif_without_pc = compute_sif_embedding_without_pc(X_sif_with_pc)

    np.save(args.x_sif_without_pc_filepath, X_sif_without_pc)

    np.save(args.x_linguistic_feature_filepath,
            compute_linguistic_features(tweets.text_body))


if __name__ =="__main__":
    main()
