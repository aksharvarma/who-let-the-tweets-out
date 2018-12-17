import os
import numpy as np
import pandas as pd
import sys

def get_basename_from_path(path):
    return os.path.basename(os.path.normpath(path))


def prepend_path(prefix_dirpath, children_dirpaths):
    return map(lambda p: os.path.join(prefix_dirpath, p),
               children_dirpaths)

def list_dirpaths(parent_dirpath):
    return filter(os.path.isdir,
                  prepend_path(parent_dirpath,
                               os.listdir(parent_dirpath)))

def create_table(model_eval_dirpath):

    evaluation = {}

    model_dirpaths = list_dirpaths(model_eval_dirpath)

    for model_dirpath in model_dirpaths:

        model = get_basename_from_path(model_dirpath)
        evaluation[model] = {}

        feature_dirpaths = list_dirpaths(model_dirpath)

        for feature_dirpath in feature_dirpaths:
            feature = get_basename_from_path(feature_dirpath)
            scores = np.loadtxt(os.path.join(feature_dirpath,
                                             "evaluation-aggregated.txt"),
                                delimiter = ",",
                                skiprows = 1)
            accuracy = scores[0,0]
            time = scores[0,4]

            evaluation[model][feature] = round(accuracy * 100, 2)

    return evaluation


def regularize_mapping(evaluation_mapping,
                       models,
                       features):
    df = pd.DataFrame()
    colnames = set()
    for model in models:
        feature_eval = evaluation_mapping[model]
        values = []
        for feature in features:
            value = feature_eval.get(feature, "???")
            if isinstance(value, int):
                value = str(value)
            values.append(value)

        df[model] = values
        values = []

    df = df.transpose()
    df.columns = features
    return df

if __name__ == "__main__":
    evaluation_mapping = create_table(sys.argv[1])
    df = regularize_mapping(evaluation_mapping,
                            ["RandomForest",
                             "Gaussian-NaiveBayes",
                             "Linear-SVM",
                             "GradientBoosting",
                             "AdaBoost",
                             "Bagging",
                             "ExtraTrees",
                             "NN",
                             "KNN-euclidean",
                             "KNN-manhattan",
                             "KNN-braycurtis",
                             "KNN-chebyshev",
                             "KNN-canberra"],
                            ["X-linguistic-feature",
                             "X-trigram",
                             "X-bigram",
                             "X-SIF-with-PC",
                             "X-SIF-without-PC"])
    print(df.to_latex())
