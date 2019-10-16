import os
import re
import sys
import glob
import random
from pprint import pprint, pformat

import config
import logging
from pprint import pprint, pformat
logging.basicConfig(format=config.FORMAT_STRING)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


import tamil

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as np

from functools import partial
from collections import namedtuple, defaultdict, Counter


from anikattu.tokenizer import word_tokenize
from anikattu.tokenstring import TokenString
from anikattu.datafeed import DataFeed
from anikattu.dataset import NLPDataset, NLPDatasetList
from anikattu.utilz import tqdm, ListTable
from anikattu.vocab import Vocab
from anikattu.utilz import Var, LongVar, init_hidden, pad_seq, flatten_dictvalues


from matplotlib import pyplot as plt

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image

Sample   =  namedtuple('Sample', ['id', 'sequence', 'label'])

class ClasswiseDataset(NLPDataset):
    def __init__(self, name, dataset, input_vocab, output_vocab):
        self.classwise_trainset, self.classwise_testset = dataset

        trainset = flatten_dictvalues(self.classwise_trainset)
        testset  = flatten_dictvalues(self.classwise_testset)

        random.shuffle(trainset)
        random.shuffle(testset)
        
        super().__init__(
            name,
            (trainset, testset),
            input_vocab,
            output_vocab
        )
        
        
def load_news_dataset(config,
                           dataset_path = '../data/dataset/news/data.csv',
                           max_sample_size=None):

    output_vocab = Counter()
    
    def load_all_data():
        skipped = 0
        samples = []

        for i, line in enumerate(tqdm(open(dataset_path).readlines())):
            try:
                _, line, label, *__ = line.split('|')
                samples.append(
                    Sample(
                    id = '{}.{}'.format(label, i),
                        sequence = line,
                        label    = label,
                    )
                )

            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except:
                skipped += 1
                log.exception(dataset_path)
                            
        print('skipped {} samples'.format(skipped))
        return samples

    samples_list = load_all_data()
    samples = defaultdict(list)
    train_samples, test_samples = {}, {}
    
    for s in samples_list:
        samples[s.label].append(s)

    for label in samples.keys():
        pivot = int( len(samples[label]) * config.CONFIG.split_ratio )
        train_samples [label] = samples[label][:pivot]
        test_samples  [label] = samples[label][pivot:]

    samples = flatten_dictvalues(samples)
        
    output_vocab.update( [s.label for s in samples]  )
    pprint([(k, output_vocab[k]) for k in sorted(output_vocab.keys())])

    return ClasswiseDataset(config.HPCONFIG.dataset_name,
                   (train_samples, test_samples),
                   Vocab(output_vocab, special_tokens=[], freq_threshold=0))



def load_movie_sentiment_dataset(config,
                           dataset_path = '../data/dataset/sentiment-analysis-movie-reviews/',
                           max_sample_size=None):

    output_vocab = Counter()
    input_vocab = Counter()
    
    def load_data(set_='train'):
        skipped = 0
        samples = []

        for i, line in enumerate(tqdm(
                open(
                    '{}/{}.tsv'.format(dataset_path, set_)
                ).readlines())):
            
            try:
                #print(line.split('\t'))
                pid, sid, line, label = line.strip().split('\t')
                samples.append(
                    Sample(
                    id = '{}.{}.{}.{}'.format(pid, sid, i, label),
                        sequence = line,
                        label    = label,
                    )
                )

            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except:
                skipped += 1
                log.exception(dataset_path)
                            
        print('skipped {} samples'.format(skipped))
        return samples

    samples_list = load_data()
    samples = defaultdict(list)
    train_samples, test_samples = {}, {}
    
    for s in samples_list:
        samples[s.label].append(s)

    for label in samples.keys():
        pivot = int( len(samples[label]) * config.CONFIG.split_ratio )
        train_samples [label] = samples[label][:pivot]
        test_samples  [label] = samples[label][pivot:]

    samples = flatten_dictvalues(samples)
        
    output_vocab.update( [s.label for s in samples]  )

    [input_vocab.update(s.sequence) for s in samples]

    pprint([(k, output_vocab[k]) for k in sorted(output_vocab.keys())])

    return ClasswiseDataset(config.HPCONFIG.dataset_name,
                            (train_samples, test_samples),
                            Vocab(input_vocab, freq_threshold=10),
                            Vocab(output_vocab, special_tokens=[], freq_threshold=0))
