PYTHON_BINPATH :=
PIP_BINPATH :=

train:
	@echo "Train Model"

install-dependencies:
	@echo "pip install commands"

raw-data:
	@mkdir -p data/raw
	@wget -O data/raw/us-political-tweets.tar.gz https://files.pushshift.io/twitter/US_PoliticalTweets.tar.gz

