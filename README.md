# Who Let the Tweets Out!

A MS/PhD Machine learning course project for Northeastern University.
This project explores author attribution problem with small texts (tweets).

## Administration

### Download raw data

Execute the following command:

```shell
$ make download-raw-data
```

### Preprocess raw data

Execute the following command:

```shell
$ make preprocess-raw-data
```

## Tasks

### Logistics
- [ ] Setup Python virtualenv
- [ ] Create dependency list (include SIF)
 
### Data
- [X] Automate download and extraction of raw data
- [X] Preprocess raw data as Pandas dataframe
- [ ] Create testing and validation sets

### Modeling
- [ ] Feature engineering (feature distribution)
- [ ] Bag of words representation of tweets
- [ ] Classification accuracy with bag of words
- [ ] Arora embedding of tweets
- [ ] Classification accuracy with arora embedding
