################################################################################
# EXECUTABLES & DEPENDENCIES
################################################################################

PYTHON_BINPATH := python
PIP_BINPATH := pip
PYTHON_DEPENDENCIES := numpy pandas nltk sklearn matplotlib

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

MODEL_DIRPATH := models

MIN_TWEET_COUNT := 500
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
                           $(MIN_TWEET_COUNT)

split-data:
	@echo "Split-data will call src/split.py to create train-test and validation data"

train-baseline-model:
	@mkdir -p $(MODEL_DIRPATH)
	@$(PYTHON_BINPATH) src/baseline_models.py \
                           $(PREPROCESSED_TWEETS_FILEPATH) \
                           $(MODEL_CHOICE) \
                           $(MODEL_DIRPATH)/$(MODEL_CHOICE).model \
                           $(MODEL_DIRPATH)/$(MODEL_CHOICE).scores

train-model:
	@echo "Train Model. Not implemented yet."

.PHONY: install-dependencies download-raw-data preprocess-raw-data train-model
