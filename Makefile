################################################################################
# EXECUTABLES & DEPENDENCIES
################################################################################

PYTHON_BINPATH := python3
PIP_BINPATH := pip3
PYTHON_DEPENDENCIES := numpy pandas nltk sklearn matplotlib progressbar2

################################################################################
# DIRECTORY & FILE PATHS
################################################################################

DATA_DIRPATH := data

RAW_DATA_DIRPATH := $(DATA_DIRPATH)/raw
RAW_DATA_URL := https://files.pushshift.io/twitter/US_PoliticalTweets.tar.gz
RAW_DATA_FILEPATH := $(RAW_DATA_DIRPATH)/us-political-tweets.tar.gz
RAW_TWEETS_FILEPATH := $(RAW_DATA_DIRPATH)/tweets.json
RAW_USERS_FILEPATH := $(RAW_DATA_DIRPATH)/users.json

PREPROCESSED_DATA_DIRPATH := $(DATA_DIRPATH)/preprocessed
PREPROCESSED_TWEETS_FILEPATH := $(PREPROCESSED_DATA_DIRPATH)/tweets.pkl
PREPROCESSED_USERS_FILEPATH := $(PREPROCESSED_DATA_DIRPATH)/users.pkl
MIN_TWEET_COUNT := 500
MAX_TWEET_COUNT := 1000000
MAX_AUTHOR_COUNT := 10

GLOVE_DATASET_NAME := glove.840B.300d
WORD_VECTOR_DATA_URL := http://nlp.stanford.edu/data/$(GLOVE_DATASET_NAME).zip
RAW_WORD_VECTOR_ARCHIVEPATH = $(RAW_DATA_DIRPATH)/word-vector.zip
RAW_WORD_VECTOR_FILEPATH := $(RAW_DATA_DIRPATH)/word-vector.txt
PREPROCESSED_WORDS_FILEPATH := $(PREPROCESSED_DATA_DIRPATH)/words.pkl
PREPROCESSED_VECTORS_FILEPATH := $(PREPROCESSED_DATA_DIRPATH)/vectors.npz

FEATURE_DIRPATH := $(DATA_DIRPATH)/features
Y_FILEPATH := $(FEATURE_DIRPATH)/Y.npy
X_BIGRAM_FILEPATH := $(FEATURE_DIRPATH)/X-bigram.npy
X_TRIGRAM_FILEPATH := $(FEATURE_DIRPATH)/X-trigram.npy
X_SIF_WIH_PC_FILEPATH := $(FEATURE_DIRPATH)/X-SIF-with-PC.npy
X_SIF_WITHOUT_PC_FILEPATH := $(FEATURE_DIRPATH)/X-SIF-without-PC.npy
X_LINGUISTIC_FEATURE_FILEPATH := $(FEATURE_DIRPATH)/X-linguistic-feature.npy


MODEL_DIRPATH := $(DATA_DIRPATH)/models
X_FILEPATH := $(X_BIGRAM_FILEPATH)
N_SPLITS := 5
RANDOM_STATE := 55861
MODEL_NAME := RandomForestClassifier
MODEL_PARAMETERS := "{'verbose': 1, 'n_jobs': -1}"
MODEL_FILEPATH := $(MODEL_DIRPATH)/$(MODEL_NAME).pkl
RAW_MODEL_SCORE_FILEPATH := $(MODEL_DIRPATH)/$(MODEL_NAME)-evaluation-raw.txt
AGGREGATED_MODEL_SCORE_FILEPATH := $(MODEL_DIRPATH)/$(MODEL_NAME)-evaluation-aggregated.txt

LOG_DIRPATH := logs
################################################################################
# RULES
################################################################################

install-dependencies:
	$(PIP_BINPATH) install --user $(PYTHON_DEPENDENCIES)

# Download and untar raw data
download-raw-data:
	@mkdir -p $(RAW_DATA_DIRPATH)
	@wget -O $(RAW_DATA_FILEPATH) $(RAW_DATA_URL)
	@tar -xvf $(RAW_DATA_FILEPATH) --directory $(RAW_DATA_DIRPATH)

# Format and remove irrelevant information
preprocess-raw-data:
	@mkdir -p $(PREPROCESSED_DATA_DIRPATH)
	@$(PYTHON_BINPATH) src/preprocess.py \
                           $(RAW_TWEETS_FILEPATH) \
                           $(RAW_USERS_FILEPATH) \
                           $(PREPROCESSED_TWEETS_FILEPATH) \
                           $(PREPROCESSED_USERS_FILEPATH) \
                           $(MIN_TWEET_COUNT) \
                           $(MAX_TWEET_COUNT) \
                           $(MAX_AUTHOR_COUNT)

# Download and unzip word vector data
download-word-vector-data:
	@mkdir -p $(RAW_DATA_DIRPATH)
	@wget -O $(RAW_WORD_VECTOR_ARCHIVEPATH) $(WORD_VECTOR_DATA_URL)
	@unzip -o $(RAW_WORD_VECTOR_ARCHIVEPATH) -d $(RAW_DATA_DIRPATH)
	@mv $(RAW_DATA_DIRPATH)/$(GLOVE_DATASET_NAME).txt $(RAW_WORD_VECTOR_FILEPATH)

# Format word vector data
preprocess-word-vector-data:
	@mkdir -p $(PREPROCESSED_DATA_DIRPATH)
	@$(PYTHON_BINPATH) src/preprocess-word-vector.py \
                           $(RAW_WORD_VECTOR_FILEPATH) \
                           $(PREPROCESSED_WORDS_FILEPATH) \
                           $(PREPROCESSED_VECTORS_FILEPATH)

extract-features:
	@mkdir -p $(FEATURE_DIRPATH)
	@$(PYTHON_BINPATH) src/extract-features.py \
											$(PREPROCESSED_TWEETS_FILEPATH) \
                      $(PREPROCESSED_WORDS_FILEPATH) \
                      $(PREPROCESSED_VECTORS_FILEPATH) \
											$(Y_FILEPATH) \
											$(X_BIGRAM_FILEPATH) \
											$(X_TRIGRAM_FILEPATH) \
											$(X_SIF_WIH_PC_FILEPATH) \
											$(X_SIF_WITHOUT_PC_FILEPATH) \
											$(X_LINGUISTIC_FEATURE_FILEPATH)


train-baseline-model:
	@mkdir -p $(MODEL_DIRPATH)
	@$(PYTHON_BINPATH) src/baseline_models.py \
                           $(PREPROCESSED_TWEETS_FILEPATH) \
                           $(MODEL_CHOICE) \
                           $(MODEL_DIRPATH)/$(MODEL_CHOICE).model \
                           $(MODEL_DIRPATH)/$(MODEL_CHOICE).scores

train-model:
	@mkdir -p $(MODEL_DIRPATH)
	@$(PYTHON_BINPATH) src/train-model.py \
                           $(X_FILEPATH) \
													 $(Y_FILEPATH) \
                           $(N_SPLITS) \
                           $(RANDOM_STATE) \
                           $(MODEL_NAME) \
	                         $(MODEL_PARAMETERS) \
                           $(MODEL_FILEPATH) \
		                       $(RAW_MODEL_SCORE_FILEPATH) \
                           $(AGGREGATED_MODEL_SCORE_FILEPATH)

train-model-sequence:
	@$(PYTHON_BINPATH) src/train-model-sequence.py \
                     $(PYTHON_BINPATH) \
                     $(FEATURE_DIRPATH) \
                     $(Y_FILEPATH) \
                     $(N_SPLITS) \
                     $(RANDOM_STATE) \
                     $(MODEL_DIRPATH)

tabulate-model-performance:
	@$(PYTHON_BINPATH) src/tabulate-model-performance.py \
                     $(MODEL_DIRPATH)

.PHONY: install-dependencies download-raw-data preprocess-raw-data download-word-vector-data preprocess-word-vector-data extract-features train-baseline-model train-model train-model-sequence tabulate-model-peformance
