from subprocess import call


MODEL_NAMES = ["GaussianNB", "RandomForestClassifier"]
X_FILEPATHS = ["X-bigram.npy", "X-trigram.npy",
               "X-linguistic-feature.npy",
               "X-SIF-with-PC.npy",
               "X-SIF-without-PC.npy"]

for model_name in MODEL_NAMES:
    for x_filepath in X_FILEPATHS:
        call([
            "ipython3",
            "src/train-model.py",
            "data/features/" + x_filepath,
            "data/features/Y.npy",
            "5",
            "55861",
            model_name,
            "{}",
            "models/" + model_name + x_filepath + ".pkl",
            "model-evaluations/" + model_name + "-" + x_filepath + "-raw.npy",
            "model-evaluations/" + model_name + "-" + x_filepath + "-aggregated.npy"])

