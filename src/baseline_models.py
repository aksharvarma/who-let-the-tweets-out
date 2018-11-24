import argparse
import numpy as np
# import pandas as pd
# import nltk
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.externals import joblib
import utils as U

class BaseLineModel():
    def __init__(self, tweets_filename=None, chosen_model='RF'):
        print('Initializing BaseLineModel: '+chosen_model)
        if tweets_filename is None:
            raise ValueError("No filename provided")
        self.tweets = U.read_data_from_file(tweets_filename)
        self.tweets = U.extract_features(self.tweets)

        self.model_options = {'RF': RandomForestClassifier,
                              'GB': GradientBoostingClassifier,
                              'Bag': BaggingClassifier,
                              'Ada': AdaBoostClassifier}

        self.chosen_model = self.model_options[chosen_model]
        self.setup_model()

    def setup_model(self):
        self.model = self.chosen_model()

    def train(self, X, y):
        self.model.fit(X, y)
        return self.predict(X)

    def predict(self, X):
        return self.model.predict(X)

    def test(self, X):
        return self.predict(X)

    def eval_preds(self, y_true, y_pred, average='weighted'):
        return (precision_score(y_true, y_pred, average=average),
                recall_score(y_true, y_pred, average=average),
                f1_score(y_true, y_pred, average=average))

    def run(self, k=10):
        X, y, x_cols = U.get_X_y(self.tweets)

        split_gen = U.get_test_train_split_generator(X, y, k=k)
        scores = []
        i = 0
        for train_inds, test_inds in split_gen:
            i += 1
            print('Fold', i, 'starting.')

            X_train, X_test = X[train_inds], X[test_inds]
            y_train, y_test = y[train_inds], y[test_inds]

            y_pred_train = self.train(X_train, y_train)
            y_pred_test = self.test(X_test)
            scores.append(self.eval_preds(y_test, y_pred_test))

        scores = np.array(scores)
        print(np.mean(scores, axis=0), np.std(scores, axis=0))
        return scores

    def save_model(self, filename):
            print('Saving model:', filename, end='. ', flush=True)
            joblib.dump(self.model, filename)
            print('Done.')

def get_arguments():
    parser = argparse.ArgumentParser(description = ("Train the BaseLine models"))

    parser.add_argument('tweet_filepath',
                        metavar = 'tweet-filepath',
                        type = str,
                        help = "pickle file containing tweet information")
    parser.add_argument('model_choice',
                        metavar = 'model-choice',
                        type = str,
                        help = "The chosen Baseline model. One of Bag (BaggingClassifier), RF (RandomForestClassifier), GB (GradientBoostingClassifier), Ada (AdaBoostClassifier).")
    parser.add_argument('model_save_filename',
                        metavar = 'model-save-filename',
                        type = str,
                        help = "File to save the trained model in.")
    parser.add_argument('score_save_filename',
                        metavar = 'score-save-filename',
                        type = str,
                        help = "File to save the scores in.")

    return parser.parse_args()

def main():
    args = get_arguments()

    model = BaseLineModel(args.tweet_filepath, args.model_choice)
    scores = model.run()
    model.save_model(args.model_save_filename)
    np.save(args.score_save_filename, scores)

main()
