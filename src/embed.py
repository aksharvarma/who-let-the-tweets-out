import numpy
import pickle
import re


class SifEmbedder(object):
    def __init__(self, words, vectors,  weight = 0.00001):
        self._words = words
        self._vectors = vectors
        self._weight = weight

    def embed(self, text):
        words = self._tokenize(text)
        word_indices = self._index(words)
        if len(word_indices) == 0:
            return numpy.zeros((1, self._vectors[1,:]))
        embedding = self._vectors[word_indices, :].sum(axis = 0)
        embedding = self._weight * embedding / len(word_indices)
        return embedding

    def _tokenize(self, text):
        words = []
        word = ""
        for c in text:
            if c.isalpha():
                word += c
            else:
                words.append(word)
                word = ""
                if not c.isspace():
                    words.append(c)
        words.append(word)
        return filter(lambda word: len(word) > 0, words)


    def _index(self, words):
        word_indices = []
        for word in words:
            index = self._words.get(word, -1)
            if index >= 0:
                word_indices.append(index)
        return numpy.array(word_indices)

