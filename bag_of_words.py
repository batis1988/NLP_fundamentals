#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 16:21:58 2024

@author: antonskvarskij
"""

import numpy as np
import pandas as pd
from typing import Union
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
    def __init__(self, corpus, vocab, max_size: Union[float, int] = None):
        self.num_objects = len(corpus)
        self.num_features = len(vocab)
        self.vocab = vocab
        self.bow = np.zeros((self.num_objects, self.num_features))
        self._fill_bow(corpus)
        self.features_names = list(self.vocab.keys())
        if isinstance(max_size, float):
            if max_size <= 1.0:
                self.limit = int(self.num_features * max_size)
            else: 
                raise ValueError("max_size = {0.0, 1.0} or not bigger than vocab length")
        if isinstance(max_size, int):
            if max_size <= self.num_features:
                self.limit = max_size
            else: 
                raise ValueError("max_size = {0.0, 1.0} or not bigger than vocab length")
        if max_size == None:
            self.limit = None    
        else:
            self.max_bow, self.features_names = self._get_maxfeatures(self.vocab, 
                                                                      self.bow)
            
        
    def _tokenize_sent(self, sentence):
        return word_tokenize(sentence)
    
    def _create_bow_vector(self, tokens):
        count_dict = defaultdict(int)
        vector = np.zeros(self.num_features)
        for token in tokens:
            count_dict[token] += 1
        for word, value in count_dict.items():
            vector[self.vocab[word]] = value
        return vector
    
    def _fill_bow(self, corpus):
        for idx, sentence in enumerate(corpus):
            self.bow[idx, :] = self._create_bow_vector(
                    self._tokenize_sent(sentence)
                )
        
    def get_bow(self):
        if self.limit == None:
            return self.bow, self.features_names
        else:
            return self.max_bow, self.features_names
            
    
    def _get_maxfeatures(self, vocab, bow):
        sorted_args = np.argsort(np.sum(bow, axis=0))[::-1].tolist()[: self.limit]
        max_features = np.take_along_axis(bow, 
                                          np.array(sorted_args)[np.newaxis, :], 
                                          axis=1)
        max_features_names = list(map(self.features_names.__getitem__, sorted_args))
        return max_features, max_features_names
    
bow = BagOfWords(text.clean_corpus, text.word2idx, max_size=0.1)

features, names = bow.get_bow()





