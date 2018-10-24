PYTHON_BINPATH := ~/projects/anaconda3/bin/python
PIP_BINPATH :=

# Download and untar raw data
download-raw-data:
	@mkdir -p data/raw
	@wget -O data/raw/us-political-tweets.tar.gz https://files.pushshift.io/twitter/US_PoliticalTweets.tar.gz
	@tar -xvf data/raw/us-political

# Split into validation set
preprocess-raw-data:
	@mkdir -p data/preprocessed
	@$(PYTHON_BINPATH) scripts/preprocess.py data/raw/tweets.json \
                                           data/raw/users.json  \
                                           data/preprocessed/tweets.pkl \
                                           data/preprocessed/users.pkl

train-model:
	@echo "Train Model"

install-dependencies:
	@echo "pip install commands"

.PHONY: download-raw-data preprocess-raw-data train-model install-dependencies
