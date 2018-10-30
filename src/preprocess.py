import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime


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
    return parser.parse_args()


def raw_json_reader(filepath, transformer):
    with open(filepath, 'r') as f:
        for line in f:
            yield transformer(json.loads(line))


def json_to_pickle(input_filepath, output_filepath, transformer):
    reader = raw_json_reader(input_filepath, transformer)
    data = pd.DataFrame(list(reader))
    data.to_pickle(output_filepath)


def process_tweet_data(raw_tweet_filepath, processed_tweet_filepath):
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

        return {
            "id": record["id"],
            "is_quote_status": record["is_quote_status"],
            "quoted_status_id": record.get("quoted_status_id", np.nan),
            "retweet_count": record["retweet_count"],
            "user_id": record["user_id"],
            "year": created_at.year,
            "month": created_at.month,
            "week": created_at.isocalendar()[1],
            "day": created_at.day,
            "hour": created_at.hour,
            "minute": created_at.minute,
            "second": created_at.second,
            "in_reply_to_status_id": record["in_reply_to_status_id"],
            "in_reply_to_user_id": record["in_reply_to_user_id"],
            "text_prologue": text[0:relevant_text_range[0]],
            "text_epilogue": text[relevant_text_range[1]:],
            "text_body": text[relevant_text_range[0]: relevant_text_range[1]],
            "user_mentions": np.array(list({d['id'] for d in user_mentions}),
                                      dtype = np.int64),
            "media": np.array(list({d['id'] for d in media}),
                              dtype = np.int64),
            "hashtags": np.array(list({d["text"] for d in hashtags}),
                                 dtype = np.str),
            "urls": np.array(list({d["expanded_url"] for d in urls}),
                             dtype = np.str),
            "symbols": np.array(list({d["text"] for d in symbols}),
                                dtype = np.str),
            "extended_media": np.array(list({d['id'] for d in extended_media}),
                                       dtype = np.int64),
            "favorite_count": record["favorite_count"]
        }

    json_to_pickle(raw_tweet_filepath,
                   processed_tweet_filepath,
                   transformer)


def process_user_data(raw_user_filepath, processed_user_filepath):
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

    print("Preprocessing tweets ...")
    process_tweet_data(args.raw_tweet_filepath,
                       args.processed_tweet_filepath)

    print("Preprocessing users ...")
    process_user_data(args.raw_user_filepath,
                      args.processed_user_filepath)


main()
