from subprocess import call


MODEL_NAMES = ["GaussianNB",
               "RandomForestClassifier",
               "KNeighborsClassifier",]

X_FILEPATHS = ["X-bigram.npy", "X-trigram.npy",
               "X-linguistic-feature.npy",
               "X-SIF-with-PC.npy",
               "X-SIF-without-PC.npy"]

MODEL_NAME_PARAMETER_MAPPING = [
    ("RandomForestClassifier", "{}"),
    ("GaussianNB", "{}"),
    ("MLPClassifier", "{'hidden_layer_sizes': (100,)}"),
    ("LinearSVC", "{'random_state': 55861}"),
    ("KNeighborsClassifier", "{'metric': 'euclidean'}"),
    ("KNeighborsClassifier", "{'metric': 'manhattan'}"),
    ("KNeighborsClassifier", "{'metric': 'chebyshev'}"),
    ("KNeighborsClassifier", "{'metric': 'minkowski'}"),
    ("KNeighborsClassifier", "{'metric': 'hamming'}"),
    ("KNeighborsClassifier", "{'metric': 'canberra'}"),
    ("KNeighborsClassifier", "{'metric': 'braycurtis'}")
]

for model_name, model_parameters in MODEL_NAME_PARAMETER_MAPPING:
    for x_filepath in X_FILEPATHS:
        call(["python3",
              "src/train-model.py",
              "data/features/" + x_filepath,
              "data/features/Y.npy",
              "5",
              "55861",
              model_name,
              model_parameters,
              "models/" + model_name + x_filepath + ".pkl",
              "model-evaluation/" + model_name + "-" + x_filepath + "-raw.npy",
              "model-evaluation/" + model_name + "-" + x_filepath + "-aggregated.npy"])

