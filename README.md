# Who Let the Tweets Out!

A Machine learning project for Northeastern University's graduate level course [CS 6140](https://sites.google.com/view/cs6140fall2018/home "CS 6140"). This project explores author attribution problem with small texts (tweets).

### Dependencies
- `make`
- `wget`
- `python3`
- `pip3`

## Administration

### The whole pipeline

To run the whole pipeline, execute the following commands:

```shell
make install-dependencies
make download-raw-data
make download-word-vector-data
make preprocess-raw-data
make preprocess-word-vector-data
make extract-features
make train-model-sequence
make tabulate-model-performance
```
Individually, these commands are explained below.


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

## Tabulate model performance

```shell
$ make tabulate-model-peformance
```
