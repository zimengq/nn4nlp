#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Zimeng Qiu <zimengq@andrew.cmu.edu>
# Licensed under the Apache License v2.0 - http://www.apache.org/licenses/

"""
This script defines some useful utilities for preprocessing data and trainning/test.
"""

import os
import torch
import gensim
import numpy as np

from matplotlib import pyplot as plt
from torchtext import data, vocab


def read_data(fname, fields):
    """
    read text data from .txt files.
    :param fname: file name
    :return: text data in list type
    """
    keys = ["label", "text"]
    with open(fname) as f:
        lines = f.readlines()
        examples = [data.Example.fromlist(line.split("|||"), fields) for line in lines]
    return examples


def load_glove_vectors(fname):
    """
    load pre-trained glove word embeddings from local disk file.
    :param fname: file name
    :return: glove word embeddings
    """
    with open(fname, 'r') as f:
        model = {}
        for line in f:
            line = line.split()
            word = line[0]
            embedding = torch.Tensor([float(val) for val in line[1:]])
            model[word] = embedding
    return model


def load_w2v_vectors(fname):
    """
    load pre-trained word2vec word embeddings from local disk file.
    :param fname: file name
    :return: word2vec word embeddings
    """
    print("Loading word2vec model from {}".format(fname))
    if not os.path.exists('model/w2v.mod'):
        model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True, limit=1000000)
        model.wv.save_word2vec_format('model/w2v.mod')
    return vocab.Vectors('model/w2v.mod')


def unk_init(dim):
    """
    Initialize out of vocabulary words as uniform
    distribution in range 0 to 1.
    :param dim: word embedding dimension
    :return: randomly initialized vector
    """
    return torch.rand(1, dim)


def visualize_data(train_data, val_data, fname):
    """
    Visualize training and validation data
    """
    fig, ax = plt.subplots()
    ax.plot(np.arange(0, len(train_data), 1), train_data, 'b')
    ax.plot(np.arange(0, len(val_data), 1), val_data, 'r')

    ax.set(xlabel='Epochs', ylabel='Error rate',
           title='Training and validation error versus training epochs.')
    ax.legend(["Training error", "Validation error"])

    for direction in ["right", "top"]:
        # hides borders
        ax.spines[direction].set_visible(False)

    if not os.path.exists('figs'):
        try:
            os.mkdir('figs')
        except OSError:
            print("Can not create directory.")

    fig.savefig("figs/" + fname + ".png")
    print("saved figure as figs/" + fname + ".png")


def write_to_file(predictions, fname):
    with open(fname, 'w') as f:
        for line in predictions:
            f.write(str(line) + '\n')
    print("Successfully write predictions to " + fname)

