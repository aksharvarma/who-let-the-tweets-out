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


def raw_json_reader(filepath):
    with open(filepath, 'r') as f:
        for line in f:
            yield json.loads(line)


def json_to_pickle(input_filepath, output_filepath):
    reader = raw_json_reader(input_filepath)
    data = pd.DataFrame(list(reader))
    data.to_pickle(output_filepath)


def process_tweet_data(raw_tweet_filepath, processed_tweet_filepath):
    json_to_pickle(raw_tweet_filepath, processed_tweet_filepath)


def process_user_data(raw_user_filepath, processed_user_filepath):
    json_to_pickle(raw_user_filepath, processed_user_filepath)


def main():
    args = get_arguments()
    print("Preprocessing tweets ...")
    process_tweet_data(args.raw_tweet_filepath,
                       args.processed_tweet_filepath)
    print("Preprocessing users ...")
    process_user_data(args.raw_user_filepath,
                      args.processed_user_filepath)

main()
