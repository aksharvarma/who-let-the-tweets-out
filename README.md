# Who Let the Tweets Out!

A MS/PhD Machine learning course project for Northeastern University.
This project explores author attribution problem with small texts (tweets).

### Dependencies
- `wget`
- `python3`
- `pip3`

## Administration

### Install Python dependencies

Execute the following command:

```shell
$ make install-dependencies
```

### Download raw data

Execute the following command:

```shell
$ make download-raw-data
```

### Download word vector data

Execute the following command:

```shell
$ make download-word-vector-data
```

### Preprocess raw data

Execute the following command:

```shell
$ make preprocess-raw-data
```

### Preprocess word vector data

Execute the following command:

```shell
$ make preprocess-word-vector-data
```

## Extract features

Execute the following command:

```shell
$ make extract-features
```

## Train all models

Execute the following command:

```shell
$ make train-model-sequence
```

## Tasks

### Logistics
- [X] Create dependency list
- [X] Understand SIF and add its [implementation](https://github.com/PrincetonML/SIF "SIF").
 
### Data
- [X] Automate download and extraction of raw data
- [X] Preprocess raw data as Pandas dataframe
- [X] Create testing and validation sets

### Modeling
- [X] Feature engineering (feature distribution)
- [X] Bag of words representation of tweets
- [X] Classification accuracy with bag of words
- [X] Arora embedding of tweets
- [X] Classification accuracy with arora embedding
