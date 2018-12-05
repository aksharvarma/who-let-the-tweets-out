import argparse
import json
import pandas as pd
import numpy as np
import progressbar
from collections import Counter
from datetime import datetime
import os
import embed
import pickle

def get_arguments():
    parser = argparse.ArgumentParser(description = ("Preprocess raw json " +
                                                    "files and store them as "
                                                    "Pandas pickle files"))
    parser.add_argument('raw_tweet_filepath',
                        metavar = 'raw-tweet-filepath',
                        type = str,
                        help = "json file containing tweet information")
    parser.add_argument('raw_user_filepath',
                        metavar = 'raw-user-filepath',
                        type = str,
                        help = "json file containing user information")
    parser.add_argument('processed_tweet_filepath',
                        metavar = 'processed-tweet-filepath',
                        type = str,
                        help = "pickle file containing processed tweet information")
    parser.add_argument('processed_user_filepath',
                        metavar = 'processed-user-filepath',
                        type = str,
                        help = "pickle file containing processed user information")
    parser.add_argument('processed_words_filepath',
                        metavar = 'processed-words-filepath',
                        type = str,
                        help = "pickle file containing processed words information")
    parser.add_argument('processed_vectors_filepath',
                        metavar = 'processed-vectors-filepath',
                        type = str,
                        help = "numpy file containing processed word vectors")
    parser.add_argument('min_tweet_count',
                        metavar = 'min-tweet-count',
                        type = int,
                        help = "minimum number of tweets a user needs to be kept in the data")
    return parser.parse_args()


def raw_json_reader(filepath, transformer):
    filesize = os.path.getsize(filepath)
    bar = progressbar.ProgressBar(max_value = filesize)
    processed_bytes = 0
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(transformer(json.loads(line)))
            processed_bytes += len(line)
            bar.update(processed_bytes)
    bar.finish()
    return data


def json_to_pickle(input_filepath, output_filepath, transformer,
                   prune_empty=False, min_tweet_count=None):
    data = pd.DataFrame(raw_json_reader(input_filepath, transformer))
    if prune_empty:
        data = prune_empty_tweets(data)
    if min_tweet_count is not None:
        data = prune_inactive_users(data, min_tweet_count=min_tweet_count)
    data.to_pickle(output_filepath)
    print('Written to:', output_filepath)


def prune_empty_tweets(tweets):
    print('\nEmpty tweets dropped:', sum(1 for tweet in tweets.text_body
                                       if len(tweet)==0))
    return tweets[tweets.text_body.map(len)!=0]

def prune_inactive_users(tweets, min_tweet_count=500):
    cntr = Counter(tweets.user_id)
    drop_user_count = sum(1 for i in cntr if cntr[i]<min_tweet_count)
    drop_tweet_count = sum(cntr[i] for i in cntr if cntr[i]<min_tweet_count)
    print('Pruning inactive users;',
          drop_user_count, 'users dropped;',
          drop_tweet_count, 'tweets dropped;')
    return tweets[tweets['user_id'].map(lambda i: cntr[i]>=min_tweet_count)]


def process_tweet_data(raw_tweet_filepath, processed_tweet_filepath,
                       preprocessed_words_filepath, preprocessed_vectors_filepath,
                       min_tweet_count=None):
    print("Preprocessing tweets...")

    preprocessed_words_file = open(preprocessed_words_filepath, "rb")
    words = pickle.load(preprocessed_words_file)
    preprocessed_words_file.close()

    preprocessed_vectors_file = open(preprocessed_vectors_filepath, "rb")
    vectors = np.load(preprocessed_vectors_file)
    preprocessed_vectors_file.close()

    embedder = embed.SifEmbedder(words, vectors)
    user_collection = []
    media_collection = []

    def transformer(record):
        tweet_column_names = ["in_reply_to_screen_name",
                              "screen_name"]

        created_at = datetime.fromtimestamp(record["created_at"])

        relevant_text_range = record["display_text_range"]
        text = record["text"]
        entities = record["entities"]

        user_mentions = entities.get("user_mentions", [])
        user_collection.extend(user_mentions)

        media = entities.get("media", [])
        media_collection.extend(media)

        hashtags = entities.get("hashtags", [])
        urls = entities.get("urls", [])
        symbols = entities.get("symbols", [])

        extended_entities = record.get("extended_entities", {})
        extended_media = extended_entities.get("media", [])
        media.extend(extended_media)

        earliest_year = 2008

        text_body = text[relevant_text_range[0]: relevant_text_range[1]]

        return {
            "id": record["id"],
            # "is_quote_status": record["is_quote_status"],
            # "quoted_status_id": record.get("quoted_status_id", np.nan),
            "retweet_count": record["retweet_count"],
            "user_id": record["user_id"],
            "year": np.int8(created_at.year - earliest_year),
            "month": np.int8(created_at.month),
            "week": np.int8(created_at.isocalendar()[1]),
            "day": np.int8(created_at.day),
            "hour": np.int8(created_at.hour),
            "minute": np.int8(created_at.minute),
            # "second": created_at.second,
            # "in_reply_to_status_id": record["in_reply_to_status_id"],
            "in_reply_to_user_id": record["in_reply_to_user_id"],
            # "text_prologue": text[0:relevant_text_range[0]],
            # "text_epilogue": text[relevant_text_range[1]:],
            "text_body": text_body,
            "avg_chars_per_word": avg_chars_per_word(text_body),
            "avg_words_per_sentence": avg_words_per_sentence(text_body),
            "num_of_chars": num_of_chars(text_body),
            "num_of_punctutations": num_of_punctutations(text_body),
            "embedding": embedder.embed(text_body),
            # "user_mentions": np.array(list({d['id'] for d in user_mentions}),
            #                           dtype = np.int64),
            # "media": len(list({d['id'] for d in media})),
            # "hashtags": np.array(list({d["text"] for d in hashtags}),
            #                      dtype = np.str),
            # "urls": np.array(list({d["expanded_url"] for d in urls}),
            #                  dtype = np.str),
            # "symbols": np.array(list({d["text"] for d in symbols}),
            #                     dtype = np.str),
            # "extended_media": np.array(list({d['id'] for d in extended_media}),
            #                            dtype = np.int64),
            "favorite_count": record["favorite_count"]
        }

    json_to_pickle(raw_tweet_filepath,
                   processed_tweet_filepath,
                   transformer,
                   prune_empty=True,
                   min_tweet_count=min_tweet_count)


def process_user_data(raw_user_filepath, processed_user_filepath):
    print("Preprocessing users...")

    def transformer(record):
        created_at = record["created_at"]

        if type(created_at) == int:
            created_at = datetime.fromtimestamp(record["created_at"])
        else:
            created_at = datetime.strptime(record["created_at"], "%a %b %d %X %z %Y")

        return {
            "id": record["id"],
            "name": record["name"],
            "screen_name": record["screen_name"],
            "year": created_at.year,
            "month": created_at.month,
            "week": created_at.isocalendar()[1],
            "day": created_at.day,
            "hour": created_at.hour,
            "minute": created_at.minute,
            "second": created_at.second,
            "followers_count": record["followers_count"],
            "friends_count": record["friends_count"]
        }

    json_to_pickle(raw_user_filepath,
                   processed_user_filepath,
                   transformer)


def main():
    args = get_arguments()

    process_tweet_data(args.raw_tweet_filepath,
                       args.processed_tweet_filepath,
                       args.processed_words_filepath,
                       args.processed_vectors_filepath,
                       min_tweet_count=args.min_tweet_count)

    process_user_data(args.raw_user_filepath,
                      args.processed_user_filepath)


main()
