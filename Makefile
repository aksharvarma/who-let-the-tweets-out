PYTHON_BINPATH := python3
PIP_BINPATH := pip3

install-dependencies:
	$(PIP_BINPATH) install --user numpy
	$(PIP_BINPATH) install --user pandas

# Download and untar raw data
download-raw-data:
	@mkdir -p data/raw
	@wget -O data/raw/us-political-tweets.tar.gz \
           https://files.pushshift.io/twitter/US_PoliticalTweets.tar.gz
	@tar -xvf data/raw/us-political-tweets.tar.gz --directory data/raw

# Split into validation set
preprocess-raw-data:
	@mkdir -p data/preprocessed
	@$(PYTHON_BINPATH) src/preprocess.py \
                     data/raw/tweets.json \
                     data/raw/users.json  \
                     data/preprocessed/tweets.pkl \
                     data/preprocessed/users.pkl

train-model:
	@echo "Train Model"

.PHONY: download-raw-data preprocess-raw-data train-model install-dependencies
