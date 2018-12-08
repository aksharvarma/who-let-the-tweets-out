from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib
import numpy as np
import argparse
from evaluation import Evaluator


def initialize_model(model_name, model_parameters):
    # model_name = args.model_name
    # model_arguments = vars(args)
    # map(model_arguments.pop, ["x_filepath",
    #                           "y_filepath",
    #                           "n_splits",
    #                           "random_state",
    #                           "model_filepath",
    #                           "model_evaluation_filepath"])
    parameters = eval(model_parameters)
    return eval(model_name + "(**parameters)")


def train_model(model, X, Y, evaluator,
                n_splits = 5, random_state = 55861):
    kfold = StratifiedKFold(n_splits = n_splits,
                            shuffle = True,
                            random_state = random_state)
    splits = kfold.split(X, Y)

    evaluator.begin()
    for train_indices, test_indices in splits:
        X_train, X_test = X[train_indices], X[test_indices]
        Y_train, Y_test = Y[train_indices], Y[test_indices]
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        evaluator.evaluate(Y_test, Y_pred)

    evaluator.finish()


def parse_arguments():
    parser = argparse.ArgumentParser(description = ("Preprocess word vector " +
                                                    "file and store it in "
                                                    "native numpy format"))
    parser.add_argument('x_filepath',
                        metavar = 'x-filepath',
                        type = str,
                        help = "npy file containing the features")
    parser.add_argument('y_filepath',
                        metavar = 'y-filepath',
                        type = str,
                        help = "npy file containing the class labels")
    parser.add_argument('n_splits',
                        metavar = 'n-splits',
                        type = int,
                        help = "number of cross validation folds")
    parser.add_argument('random_state',
                        metavar = 'random-state',
                        type = int,
                        help = "random state for cross validation shuffling")
    parser.add_argument("model_name",
                        metavar = "model-name",
                        type = str,
                        help = "name of the model sklearn model class")
    parser.add_argument("model_parameters",
                        metavar = "model-parameters",
                        type = str,
                        help = "parameters passed to the model being trained")
    parser.add_argument("model_filepath",
                        metavar = "model-filepath",
                        type = str,
                        help = "file containing the trained model")
    parser.add_argument("model_evaluation_filepath",
                        metavar = "model-evaluation-filepath",
                        type = str,
                        help = "file containing model evaluation")

    return parser.parse_args()


def main():
    args = parse_arguments()

    print(args)

    model = initialize_model(args.model_name,
                             args.model_parameters)

    X = np.load(args.x_filepath)

    Y = np.load(args.y_filepath)

    train_model(model, X, Y,
                Evaluator(),
                args.n_splits,
                args.random_state)

#    evaluator.save(args.model_evaluation_filepath)

    joblib.dump(model, args.model_filepath)


if __name__ =="__main__":
    main()

