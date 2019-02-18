# CNN Based Text Classifier

Author: Zimeng Qiu <zimengq@andrew.cmu.edu>

This is a CNN based text classifier for CMU 11-747 Neural Networks for NLP assignment 1. I reproduced the work by Yoon Kim in [Convolutional Neural Networks for Sentence Classification](https://www.aclweb.org/anthology/D14-1181), achieved around 81% accuracy on [course given dataset](http://phontron.com/class/nn4nlp2019/assignments.html).

## Requirements

These scripts works best in both Python 3 and Python 2, below are python packages required.

- torch==1.0.1
- torchtext=0.3.1
- nltk=3.4
- gensim=3.7.1
- matplotlib=3.0.2

## How to Run

Install requirements

```
pip install -r requirements.txt
```

Train and test a custom multi-layer neural network

```
usage: train.py [-h] [--embedding {w2v,glove}] [--filter_num FILTER_NUM]
                [--filter_sizes FILTER_SIZES [FILTER_SIZES ...]]
                [--multichannel {1,2}] [--dropout DROPOUT] [--epoch EPOCH]
                [--optim OPTIM] [--debug DEBUG]
```

Use -h for usage information:

```
python run.py -h
```

And it will show that

```
CNN parameters.

optional arguments:
  -h, --help            show this help message and exit
  --embedding {w2v,glove}
                        pre-trained word embeddings <w2v|glove>, default is
                        w2v.
  --filter_num FILTER_NUM
                        number of filters for each filter size
  --filter_sizes FILTER_SIZES [FILTER_SIZES ...]
                        list type, sizes of filters.
  --multichannel {1,2}  number of input channels, default is 2, use one static
                        and one non-static word vectors matrix
  --dropout DROPOUT     dropout rate, float number from 0 to 1.
  --epoch EPOCH         trainning epochs.
  --optim OPTIM         optimizer, Adadelta, Adam or SGD
  --debug DEBUG         debugging mode, only use dev set, not enabled if set
                        0.

```

An example:

```
python run.py --filter_num 64 --filter_sizes 3 4 5 --optim sgd
```
