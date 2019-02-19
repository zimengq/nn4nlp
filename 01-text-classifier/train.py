#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Zimeng Qiu <zimengq@andrew.cmu.edu>
# Licensed under the Apache License v2.0 - http://www.apache.org/licenses/

"""
This script train a text classifier with pytorch.
"""

# TODO: keep one channel in CNN static

import os
import logging
import torch
import random
import argparse
import codecs
import pandas as pd

from torchtext import data, vocab
from nltk.corpus import stopwords
from utils import *
from preprocessor import Preprocessor

SEED = 11747
STOP_WORDS = stopwords.words('english')
BATCH_SIZE = 64
TEXT = data.Field(
    sequential=True,
    use_vocab=True,
    lower=True,
    tokenize=Preprocessor(stopwords=STOP_WORDS, lemmatization=True).tokenizer,
)
LABEL = data.Field(sequential=False, unk_token="UNK")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FIELDS = [('label', LABEL), ('text', TEXT)]


class ConvNet(torch.nn.Module):
    """
    Convolution Neural Netwrok for text classification task
    """
    def __init__(self, embeddings, embedding_dim, output_dim, vocab_size, filter_num,
                 filter_sizes, multichannel=2, dropout=0.5):
        """
        constructor
        :param embeddings: pre-trained word embeddings
        :param embedding_dim: dimension of word embeddings
        :param output_dim: output dimension
        :param vocab_size: vocabulary size
        :param filter_num: number of convolution filters
        :param filter_sizes: convolution filter sizes
        :param multichannel: input channel number
        :param dropout: dropout rate
        """
        super(ConvNet, self).__init__()
        self.channels = multichannel

        # use pre-trained word embeddings
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = torch.nn.Parameter(embeddings)

        # 1d convolution layer applied on concatenated h words, e.g. c_i = f(w*x_{i:i+h-1}+b)
        # there're total filter_num filters, each varies in window size
        self.convs = torch.nn.ModuleList([
                torch.nn.Conv1d(
                    in_channels=embedding_dim, out_channels=filter_num,
                    kernel_size=size, stride=1
                ) for size in filter_sizes])

        self.pooling = torch.nn.AdaptiveMaxPool1d(1)
        self.fc = torch.nn.Linear(filter_num*len(filter_sizes), output_dim, bias=True)
        self.dropout = torch.nn.Dropout(dropout)
        # self.softmax = torch.nn.LogSoftmax(dim=1)

        # Initializing the projection layer
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        """
        forward propagation
        """
        # multichannel, stacking two word matrix
        if self.channels == 2:
            x = torch.stack((x, x), dim=0)
        # else:
        #     x = x.unsqueeze(0)
        # x = x.permute(2, 0, 1)
        x = x.permute(1, 0)

        # forward propagation
        embedding_layer = self.embedding(x).permute(0, 2, 1)
        conv_layer = [torch.nn.functional.relu(conv(embedding_layer)) for conv in self.convs]
        torch.save(conv_layer, 'model/convs')
        pooling_layer = [self.pooling(conv) for conv in conv_layer]
        fc_layer = self.fc(self.dropout(torch.cat(pooling_layer, dim=1)).squeeze(2))

        return fc_layer


def train(model, train_iter, optimizer, criterion):
    """
    train model
    :param model: trained model
    :param train_iter: train iterator
    :return: train loss
    """
    avg_loss = 0
    correct = 0
    total_sent = 0
    model.train()
    # training process
    for batch in train_iter:
        total_sent += len(batch)
        # zero gradients
        optimizer.zero_grad()
        # predict label
        prediction_result = model(batch.text)

        # compute and print loss
        loss = criterion(prediction_result, batch.label)
        # print("Training: batch: {}, {} sentences, loss: {:.3f}, total {} sentences trianed".format(
        #     iter, len(batch), loss.item(), total))

        # perform a backward pass, and update the weights.
        loss.backward()
        optimizer.step()

        # update loss and acc
        avg_loss += loss.item()
        correct += torch.eq(torch.argmax(prediction_result, dim=1), batch.label).sum().item()

    return avg_loss / len(train_iter), float(correct)/float(total_sent)


def evaluate(model, iterator, criterion):
    """
    evaluate trained model.
    :param model: trained model
    :param iterator: test/valid iterator
    :return: test/valid loss
    """
    avg_loss = 0
    total_sent = 0
    correct = 0
    model.eval()
    pred_labels = []

    # evaluate process
    # deactivate gradient engine
    # no back propagation
    with torch.no_grad():
        for batch in iterator:
            total_sent += len(batch)
            # predict label
            prediction_result = model(batch.text)
            # compute and print loss
            loss = criterion(prediction_result, batch.label)
            # print("Evaluating: batch: {}, {} sentences, loss: {:.3f}, total {} sentences trianed".format(
            #     iter, len(batch), loss.item(), total_sent))
            # update loss and acc
            avg_loss += loss.item()
            pred = torch.argmax(prediction_result, dim=1)
            pred_labels.extend(pred.tolist())
            correct += torch.eq(pred, batch.label).sum().item()

    return avg_loss/len(iterator), float(correct)/float(total_sent), [LABEL.vocab.itos[label] for label in pred_labels]


def test(model, iterator):
    """
    predict labels using trained model.
    :param model: trained model
    :param iterator: test/valid iterator
    :return: test/valid loss
    """
    model.eval()
    pred_labels = []

    # evaluate process
    # deactivate gradient engine
    # no back propagation
    with torch.no_grad():
        for batch in iterator:
            # predict label
            prediction_result = model(batch.text)
            pred = torch.argmax(prediction_result, dim=1).tolist()
            pred_labels.extend(pred)

    return [LABEL.vocab.itos[label] for label in pred_labels]


def compute_saliency_map(model, sample_doc, label, _dir, fname, criterion, itos):
    """
    This function is modified from https://github.com/EdGENetworks/anuvada
    """
    model.zero_grad()
    model.eval()
    scores = model.forward(sample_doc.unsqueeze(1))
    loss = criterion(scores, label)
    loss.backward()
    grad_of_param = {}
    for name, parameter in model.named_parameters():
        if 'embed' in name:
            grad_of_param[name] = parameter.grad
    grad_embed = grad_of_param['embedding.weight']
    sensitivity = torch.pow(grad_embed, 2).mean(dim=1)
    sensitivity = list(sensitivity.data.cpu().numpy())
    i2w = [itos[zz] for zz in sample_doc.data.tolist()]
    activations = [sensitivity[yy] for yy in sample_doc.data.tolist()]
    df = pd.DataFrame({'word': i2w, 'senstivity': activations})
    words = df.word.values
    values = df.senstivity.values
    if not os.path.exists(_dir):
        try:
            os.mkdir(_dir)
        except OSError:
            print("Can not create directory.")
    with codecs.open(os.path.join(_dir, fname), "w", encoding="utf-8") as html_file:
        for word, alpha in zip(words, values / values.max()):
            if not word == '<pad>':
                html_file.write('<font style="background: rgba(255, 255, 0, %f)">%s</font>\n' % (alpha, word))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description='CNN parameters.')
    parser.add_argument('--embedding', help='pre-trained word embeddings <w2v|glove>, default is w2v.',
                        default='w2v', choices=['w2v', 'glove', 'fasttext'])
    parser.add_argument('--filter_num', help='number of filters for each filter size', default=100, type=int)
    parser.add_argument('--filter_sizes', help='list type, sizes of filters.', default=[3, 4, 5], nargs='+', type=int)
    parser.add_argument(
        '--multichannel',
        help='number of input channels, default is 2, use one static and one non-static word vectors matrix',
        type=int, choices=[1, 2], default=1
    )
    parser.add_argument('--dropout', help='dropout rate, float number from 0 to 1.', default=0.5, type=float)
    parser.add_argument('--epoch', help='trainning epochs.', default=10, type=int)
    parser.add_argument('--optim', help='optimizer, Adadelta, Adam or SGD', default='adadelta')
    parser.add_argument('--debug', help='debugging mode, only use dev set, not enabled if set 0.', default=1, type=int)

    args = parser.parse_args()

    # load pre-trained word embeddings
    if args.embedding.lower() == 'glove':
        pretrained_embeddings = vocab.GloVe(name='42B')
    elif args.embedding.lower() == 'fasttext':
        pretrained_embeddings = vocab.FastText(max_vectors=500000)
    else:
        if not os.path.exists('model/GoogleNews-vectors-negative300.bin.gz'):
            os.system('wget https://drive.google.com/uc?export=download&confirm=irnl&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM')
        pretrained_embeddings = load_w2v_vectors('model/GoogleNews-vectors-negative300.bin.gz')

    # prepare dataset
    logger.info('Preparing dataset...')
    train_data = data.Dataset(read_data('data/topicclass/topicclass_train.txt', FIELDS), fields=FIELDS)
    valid_data = data.Dataset(read_data('data/topicclass/topicclass_valid.txt', FIELDS), fields=FIELDS)
    test_data = data.Dataset(read_data('data/topicclass/topicclass_test.txt', FIELDS), fields=FIELDS)
    print("="*80)
    print(len(train_data), len(valid_data), len(test_data))
    print("="*80)

    if args.debug == 0:
        # normal mode, load all training data
        TEXT.build_vocab(train_data)
        LABEL.build_vocab(train_data)
        train_iter, valid_iter, test_iter = data.Iterator.splits(
            (train_data, valid_data, test_data),
            batch_size=BATCH_SIZE,
            device=DEVICE,
            sort=False,
            # sort_key=lambda x: len(x.text),
            sort_within_batch=False,
            repeat=False,
            shuffle=True
        )
    else:
        # debugging mode, only use subset of training data
        dev_data, other_data = train_data.split(split_ratio=0.1, random_state=random.seed(SEED))
        TEXT.build_vocab(dev_data)
        LABEL.build_vocab(dev_data)
        train_iter, valid_iter, test_iter = data.Iterator.splits(
            (dev_data, valid_data, test_data),
            batch_size=BATCH_SIZE,
            device=DEVICE,
            sort=False,
            # sort_key=lambda x: len(x.text),
            sort_within_batch=False,
            repeat=False,
            shuffle=True
        )
    logger.info('Done!')
    TEXT.vocab.set_vectors(pretrained_embeddings.stoi, pretrained_embeddings.vectors, pretrained_embeddings.dim)

    valid_iter, test_iter = data.Iterator.splits(
        (valid_data, test_data),
        batch_size=BATCH_SIZE,
        device=DEVICE,
        sort=False,
        sort_within_batch=False,
        repeat=False,
        shuffle=False
    )

    # Construct model
    logger.info('Start training model...')
    model = ConvNet(embeddings=TEXT.vocab.vectors, embedding_dim=TEXT.vocab.vectors.shape[1],
                    output_dim=len(LABEL.vocab), vocab_size=len(TEXT.vocab), filter_num=args.filter_num,
                    filter_sizes=args.filter_sizes, multichannel=args.multichannel, dropout=args.dropout)
    model = model.to(DEVICE)

    max_acc = 0
    train_loss_record = []
    valid_loss_record = []
    train_acc_record = []
    valid_acc_record = []
    best_valid_predictions = []
    test_predictions = []
    # Construct loss function and Optimizer.
    criterion = torch.nn.CrossEntropyLoss()
    if args.optim.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters())
    elif args.optim.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    elif args.optim.lower() == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters())
    else:
        optimizer = torch.optim.Adadelta(model.parameters())
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.5, patience=2)
    # start training model
    for epoch in range(args.epoch):
        train_loss, train_acc = train(model, train_iter, optimizer, criterion)
        valid_loss, valid_acc, valid_predictions = evaluate(model, valid_iter, criterion)
        train_loss_record.append(train_loss)
        train_acc_record.append(train_acc)
        valid_loss_record.append(valid_loss)
        valid_acc_record.append(valid_acc)
        # scheduler.step(valid_acc)
        logger.info('Epoch: {}, train loss: {:.4f}, train accuracy: {:.4f} valid loss: {:.4f}, '
                    'valid accuracy: {:.4f}'.format(epoch+1, train_loss, train_acc, valid_loss, valid_acc))
        if valid_acc > max_acc:
            torch.save(model.state_dict(), 'model/model.pt')
            max_acc = valid_acc
            logger.info("Saved new model at model/model.pt.")
            best_valid_predictions = valid_predictions
            test_predictions = test(model, test_iter)

    sample = "i love natural language processing very much".split()
    sample_doc = torch.tensor([TEXT.vocab.stoi[char] for char in sample])
    label = torch.tensor(LABEL.vocab.stoi["Natural sciences "]).unsqueeze(0)
    compute_saliency_map(model, sample_doc, label, 'figs', 'saliency.html', criterion, pretrained_embeddings.itos)
    write_to_file(best_valid_predictions, 'val_pred.txt')
    write_to_file(test_predictions, 'test_pred.txt')
    visualize_data(train_acc_record, valid_acc_record, 'acc')
    visualize_data(train_loss_record, valid_loss_record, 'loss')
