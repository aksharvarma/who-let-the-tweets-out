import argparse
import json
import pandas as pd
import numpy as np
import progressbar
from collections import Counter
from datetime import datetime
import os
import pickle
import sys

def parse_arguments():
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
    parser.add_argument('min_tweet_count',
                        metavar = 'min-tweet-count',
                        type = int,
                        default = 0,
                        help = "minimum number of tweets a user needs to be kept in the data")
    parser.add_argument('max_tweet_count',
                        metavar = 'max-tweet-count',
                        type = int,
                        default = sys.maxsize,
                        help = "maximum number of tweets a user needs to be kept in the data")
    parser.add_argument('max_author_count',
                        metavar = 'max_author_count',
                        type = int,
                        default = sys.maxsize,
                        help = "maximum number of authors (in descending order of number of tweets) that will be kept in the data")

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


def json_to_pickle(input_filepath,
                   output_filepath,
                   transformer,
                   reducer = lambda i : i):
    data = reducer(pd.DataFrame(raw_json_reader(input_filepath, transformer)))
    data.to_pickle(output_filepath)
    print('Written to:', output_filepath)



def extract_tweet_information(record):
    tweet_column_names = ["in_reply_to_screen_name",
                          "screen_name"]

    created_at = datetime.fromtimestamp(record["created_at"])

    relevant_text_range = record["display_text_range"]
    text = record["text"]
    entities = record["entities"]

    user_mentions = entities.get("user_mentions", [])
    # user_collection.extend(user_mentions)

    media = entities.get("media", [])
    # media_collection.extend(media)

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


def filter_tweet_dataset(tweets,
                         min_tweet_count,
                         max_tweet_count,
                         max_author_count):

    def fst(pair): return pair[0]

    def snd(pair): return pair[1]

    print("Removing records with empty tweets: ", end = "")
    pruned_tweets = tweets[tweets.text_body.map(len)!=0]
    print(tweets.shape[0] - pruned_tweets.shape[0], "records removed")

    print("Selecting users with ", min_tweet_count, " <= tweets <= ", max_tweet_count)

    id_count_pairs =  Counter(pruned_tweets.user_id).items()
    filtered_id_count_pairs = filter(
        lambda p: min_tweet_count <= p[1] <= max_tweet_count,
        id_count_pairs)
    sorted_id_count_pairs = sorted(filtered_id_count_pairs,
                                   key = snd,
                                   reverse = True)
    selected_id_count_pairs = sorted_id_count_pairs[0:max_author_count]

    user_ids = set(map(fst, selected_id_count_pairs))
    tweet_counts = map(snd, selected_id_count_pairs)

    print("Original Tweet Count:  ", tweets.shape[0])
    print("Pruned Tweet Count:    ", pruned_tweets.shape[0])
    print("Retained Tweet Count:  ", sum(tweet_counts))

    print("Original Author Count: ", tweets.user_id.unique().shape[0])
    print("Pruned Author Count:   ", len(id_count_pairs))
    print("Retained Author Count: ", len(user_ids))

    return pruned_tweets[pruned_tweets["user_id"].map(lambda i: i in user_ids)]


def process_tweet_data(raw_tweet_filepath,
                       processed_tweet_filepath,
                       min_tweet_count,
                       max_tweet_count,
                       max_author_count):

    print("Preprocessing tweets...")
    data = pd.DataFrame(raw_json_reader(raw_tweet_filepath, extract_tweet_information))
    data = filter_tweet_dataset(data, min_tweet_count, max_tweet_count, max_author_count)
    data.to_pickle(processed_tweet_filepath)
    print('Written to:', processed_tweet_filepath)


def extract_user_information(record):
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

def process_user_data(raw_user_filepath, processed_user_filepath):
    print("Preprocessing users...")
    print("Preprocessing tweets...")
    data = pd.DataFrame(raw_json_reader(raw_user_filepath, extract_user_information))
    data.to_pickle(processed_user_filepath)
    print('Written to:', processed_user_filepath)


def main():
    args = parse_arguments()

    process_tweet_data(args.raw_tweet_filepath,
                       args.processed_tweet_filepath,
                       min_tweet_count = args.min_tweet_count,
                       max_tweet_count = args.max_tweet_count,
                       max_author_count = args.max_author_count)

    process_user_data(args.raw_user_filepath,
                      args.processed_user_filepath)


main()
