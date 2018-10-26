import argparse
import json
import pandas as pd

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


def raw_json_reader(filepath, columns):
    with open(filepath, 'r') as f:
        for line in f:
            record = json.loads(line)
            yield {key: record.get(key, None) for key in columns}


def json_to_pickle(input_filepath, output_filepath, columns):
    reader = raw_json_reader(input_filepath, columns)
    data = pd.DataFrame(list(reader))
    data.to_pickle(output_filepath)


def process_tweet_data(raw_tweet_filepath, processed_tweet_filepath, columns):
    json_to_pickle(raw_tweet_filepath, processed_tweet_filepath, columns)


def process_user_data(raw_user_filepath, processed_user_filepath, columns):
    json_to_pickle(raw_user_filepath, processed_user_filepath, columns)


def main():
    args = get_arguments()


    tweet_column_names = ["created_at",
                          "display_text_range",
                          "entities",
                          "extended_entities",
                          "favorite_count",
                          "id",
                          "in_reply_to_screen_name",
                          "in_reply_to_status_id",
                          "in_reply_to_user_id",
                          "is_quote_status",
                          "quoted_status_id",
                          "retweet_count",
                          "screen_name",
                          "text",
                          "user_id"]

    print("Preprocessing tweets ...")
    process_tweet_data(args.raw_tweet_filepath,
                       args.processed_tweet_filepath,
                       tweet_column_names)

    user_column_names = ["created_at",
                         "followers_count",
                         "friends_count",
                         "id",
                         "name",
                         "screen_name"]

    print("Preprocessing users ...")
    process_user_data(args.raw_user_filepath,
                      args.processed_user_filepath,
                      user_column_names)

main()
