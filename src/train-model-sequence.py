from subprocess import call
import os
import argparse


def train_model_sequence(model_name_class_parameter_mapping,
                         features,
                         python_binpath,
                         x_dirpath,
                         y_filepath,
                         n_splits,
                         random_state,
                         model_dirpath):

    for model_name, model_class, model_parameters in model_name_class_parameter_mapping:
        for feature in features:
            x_filepath = os.path.join(x_dirpath, feature + ".npy")
            model_feature_dirpath = os.path.join(model_dirpath, model_name, feature)
            os.makedirs(model_feature_dirpath)

            model_filepath = os.path.join(model_feature_dirpath, "model.pkl")
            raw_eval_filepath = os.path.join(model_feature_dirpath, "evaluation-raw.txt")
            agg_eval_filepath = os.path.join(model_feature_dirpath, "evaluation-aggregated.txt")

            command = [python_binpath,
                       "src/train-model.py",
                       x_filepath,
                       y_filepath,
                       str(n_splits),
                       str(random_state),
                       model_class,
                       str(model_parameters),
                       model_filepath,
                       raw_eval_filepath,
                       agg_eval_filepath]

            retcode = call(command)

            if retcode != 0:
                print("Return code != 0, exiting ...")
                exit(0)


def parse_arguments():
    parser = argparse.ArgumentParser(description = ("Train a sequence of models " +
                                                    "on the given set of features " +
                                                    "and evaluate their performance"))
    parser.add_argument('python_binpath',
                        metavar = 'python-binpath',
                        type = str,
                        help = "python binary with which to train the models")
    parser.add_argument('x_dirpath',
                        metavar = 'x-dirpath',
                        type = str,
                        help = "directory containing the features")
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
    parser.add_argument("model_dirpath",
                        metavar = "model-dirpath",
                        type = str,
                        help = "directory containing the trained model hierarchy")

    return parser.parse_args()


def main():
    args = parse_arguments()

    features = ["X-linguistic-feature",
                "X-trigram",
                "X-bigram",
                "X-SIF-with-PC",
                "X-SIF-without-PC"]

    model_name_class_parameter_mapping = [
        ("RandomForest", "RandomForestClassifier", {}),
        ("Gaussian-NaiveBayes", "GaussianNB", {}),
        ("Linear-SVM", "LinearSVC", {'random_state': 55861}),
        ("AdaBoost", "AdaBoostClassifier", {}),
        ("Bagging", "BaggingClassifier", {}),
        ("GradientBoosting", "GradientBoostingClassifier", {}),
        ("ExtraTrees", "ExtraTreesClassifier", {}),
        ("KNN-euclidean", "KNeighborsClassifier", {'metric': 'euclidean'}),
        ("KNN-manhattan", "KNeighborsClassifier", {'metric': 'manhattan'}),
        ("KNN-chebyshev", "KNeighborsClassifier", {'metric': 'chebyshev'}),
        ("KNN-canberra", "KNeighborsClassifier", {'metric': 'canberra'}),
        ("KNN-braycurtis", "KNeighborsClassifier", {'metric': 'braycurtis'}),
        ("NN", "MLPClassifier", {'hidden_layer_sizes': (100,), "max_iter": 1000})
    ]

    train_model_sequence(model_name_class_parameter_mapping,
                         features,
                         args.python_binpath,
                         args.x_dirpath,
                         args.y_filepath,
                         args.n_splits,
                         args.random_state,
                         args.model_dirpath)



if __name__ == "__main__":
    main()
