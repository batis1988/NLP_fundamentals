#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 16:21:58 2024

@author: antonskvarskij
"""

import numpy as np
import pandas as pd
from typing import Union, List
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from text_preprocessing import TextPreprocesser

nltk.download('stopwords')

DATA_LINK = "./data/Twitterdatainsheets.csv"

data = pd.read_csv(DATA_LINK)
corpus = data[' text'].dropna().drop_duplicates().values
stopwords = nltk.corpus.stopwords.words('english')

text = TextPreprocesser(raw_corpus=corpus, stopwords=stopwords)


class BagOfWords:
    def __init__(self, max_size: Union[float, int, None] = None):
        self.is_fitted = False
        if isinstance(max_size, float):
            assert max_size <= 1.0, "max_size = {0.0, 1.0} or not bigger than vocab length"
            self.limit = max_size
        elif isinstance(max_size, int):
            self.limit = max_size
        if max_size is None:
            self.limit = None

    @staticmethod
    def _tokenize_sent(sentence):
        return word_tokenize(sentence)

    def _create_bow_vector(self, tokens):
        count_dict = defaultdict(int)
        vector = np.zeros(self.num_features)
        for token in tokens:
            count_dict[token] += 1
        for word, value in count_dict.items():
            if word in self.vocab:
                vector[self.vocab[word]] = value
            else:
                pass
        return vector

    def _fill_bow(self, corpus):
        for idx, sentence in enumerate(corpus):
            self.bow[idx, :] = self._create_bow_vector(
                self._tokenize_sent(sentence)
            )

    def _get_bow(self):
        if self.limit is None:
            return self.bow, self.features_names
        else:
            return self.max_bow, self.features_names

    def _get_maxfeatures(self, vocab, bow):
        sorted_args = np.argsort(np.sum(bow > 0, axis=0))[::-1].tolist()[: self.limit]
        max_features = np.take_along_axis(bow,
                                          np.array(sorted_args)[np.newaxis, :],
                                          axis=1)
        max_features_names = list(map(self.features_names.__getitem__, sorted_args))
        return max_features, max_features_names

    def fit(self, corpus: List[str], vocab=None) -> None:
        self.num_objects = len(corpus)

        if vocab is None:
            vocab = {}
            idx = 0
            for sentence in corpus:
                for word in self._tokenize_sent(sentence):
                    if word not in vocab:
                        vocab[word] = idx
                        idx += 1
            self.vocab = vocab
        else:
            self.vocab = vocab

        self.num_features = len(vocab)
        self.bow = np.zeros((self.num_objects, self.num_features))
        self._fill_bow(corpus)
        self.features_names = list(self.vocab.keys())
        if isinstance(self.limit, float):
            self.limit = int(self.num_features * self.limit)
        elif self.limit is not None:
            self.max_bow, self.max_features_names = self._get_maxfeatures(self.vocab,
                                                                          self.bow)
        self.is_fitted = True

    def fit_transform(self, corpus, vocab=None):
        self.fit(corpus, vocab)
        return self._get_bow()

    def transform(self, corpus: List[str]):
        if self.is_fitted:
            bow = np.zeros((len(corpus), len(self.vocab)))
            tokens = [self._tokenize_sent(sent) for sent in corpus]
            for idx, sentence in enumerate(tokens):
                bow[idx, :] = self._create_bow_vector(sentence)
            if self.limit is not None:
                max_bow, names = self._get_maxfeatures(self.vocab, bow)
                return max_bow, names
            if self.limit == None:
                return bow, self.features_names
        else:
            raise AttributeError("Not Fitted yet")


bow = BagOfWords(max_size=50)
features, names = bow.fit_transform(text.clean_corpus)
