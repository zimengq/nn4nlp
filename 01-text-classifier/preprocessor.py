#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Zimeng Qiu <zimengq@andrew.cmu.edu>
# Licensed under the Apache License v2.0 - http://www.apache.org/licenses/

"""
This script defines a class for text data preprocessing
"""

import string

from nltk.stem.wordnet import WordNetLemmatizer


class Preprocessor(object):
    """
    class for preprocessing text data in pipeline
    """
    def __init__(self, stopwords, lemmatization=False):
        self.stopwords = stopwords
        self.lem = lemmatization
        self.words = []

    def tokenizer(self, sentence):
        for punc in string.punctuation:
            sentence = sentence.replace(punc, '')
        sentence = sentence.split()
        self.words = [word.lower() for word in sentence if word.lower() not in self.stopwords and 2 < len(word) < 20
                      and all(char.isalpha() or char in ["\"", "\'", "-"] for char in word)]
        if self.lem is True:
            self.words = [WordNetLemmatizer().lemmatize(word) for word in self.words]
        return self.words






