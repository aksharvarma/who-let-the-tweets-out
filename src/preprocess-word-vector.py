import numpy
import argparse
import progressbar
import pickle

def process_word_vector_data(word_vector_filepath,
                             processed_words_filepath,
                             processed_vectors_filepath):

    print(f"Loading word vector file '{word_vector_filepath}' ...")

    word_vector_file = open(word_vector_filepath, "r")
    word_vector_data = word_vector_file.readlines()
    word_vector_file.close()

    print(f"Processing word vector data ...")
    row_count = len(word_vector_data)
    col_count = len(word_vector_data[0].split(" ")) - 1

    # We will save space if we use float32 but that can
    # lead to loss of precision due to rounding errors.
    # It may also be the case that float32 is precise enough
    # for our use cases and float64 inflates the data size
    # too much.
    # The correct datatype choice is not clear.
    vector_data = numpy.empty((row_count, col_count), dtype = numpy.float64)

    word_data = {}

    bar = progressbar.ProgressBar(max_value = row_count)

    for row_index in range(0, row_count):
        row = word_vector_data[row_index].split(" ")
        vector_data[row_index] = numpy.float64(row[1:])
        word_data[row[0]] = row_index
        if row_index % 50000 == 0 or row_index == row_count - 1:
            bar.update(row_index)

    print(f"Saving processed vector file '{processed_vectors_filepath}' ...")
    processed_vectors_file = open(processed_vectors_filepath, "wb")
    numpy.save(processed_vectors_file, vector_data)
    processed_vectors_file.close()

    print(f"Saving processed word file '{processed_words_filepath}' ...")
    processed_words_file = open(processed_words_filepath, "wb")
    pickle.dump(word_data, processed_words_file)
    processed_words_file.close()


def get_arguments():
    parser = argparse.ArgumentParser(description = ("Preprocess word vector " +
                                                    "file and store it in "
                                                    "native numpy format"))
    parser.add_argument('word_vector_filepath',
                        metavar = 'word-vector-filepath',
                        type = str,
                        help = "text file containing word embedding")
    parser.add_argument('processed_words_filepath',
                        metavar = 'processed-words-filepath',
                        type = str,
                        help = "pickle file containing word to index mapping")
    parser.add_argument('processed_vectors_filepath',
                        metavar = 'processed-vectors-filepath',
                        type = str,
                        help = "npz file containing embedding of words by their index")

    return parser.parse_args()


def main():
    args = get_arguments()

    process_word_vector_data(args.word_vector_filepath,
                             args.processed_words_filepath,
                             args.processed_vectors_filepath)


if __name__ == "__main__":
    main()
