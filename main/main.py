import os
import re
import sys
import json
import time
import random
from pprint import pprint, pformat

sys.path.append('..')
import config
from anikattu.logger import CMDFilter
import logging
logging.basicConfig(format=config.FORMAT_STRING)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
import sys


from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
import torch

from anikattu.utilz import initialize_task

from model import model

import utilz

from functools import partial


from anikattu.tokenizer import word_tokenize
from anikattu.tokenstring import TokenString
from anikattu.datafeed import DataFeed
from anikattu.dataset import NLPDataset as Dataset, NLPDatasetList as DatasetList
from anikattu.utilz import tqdm, ListTable, dump_vocab_tsv, dump_cosine_similarity_tsv
from anikattu.vocab import Vocab
from anikattu.utilz import Var, LongVar, init_hidden, pad_seq


import importlib


SELF_NAME = os.path.basename(__file__).replace('.py', '')

import sys
import pickle
import argparse
from matplotlib import pyplot as plt
plt.style.use('ggplot')
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SkipGram model for training tamil language model')
    parser.add_argument('-p','--hpconfig',
                        help='path to the hyperparameters config file',
                        default='hpconfig.py', dest='hpconfig')

    parser.add_argument('-d','--results-dir',
                        help='path to the results root directory',
                        default='run00', dest='results_dir')
    
    parser.add_argument('--log-filters',
                        help='log filters',
                        dest='log_filter')

    subparsers = parser.add_subparsers(help='commands')
    train_parser = subparsers.add_parser('train', help='starts training')
    train_parser.add_argument('--train', default='train', dest='task')
    train_parser.add_argument('--mux', action='store_true', default=False, dest='mux')

    
    dump_cosine_similarity_parser = subparsers.add_parser('dump-cosine-similarity',
                                                          help='dumps the cosine-similarity into two tsv file')
    dump_cosine_similarity_parser.add_argument('--dump-cosine-similarity',
                                               default='dump-cosine-similarity',
                                               dest='task')

    
    predict_parser = subparsers.add_parser('predict',
                                help='''starts a cli interface for running predictions 
                                in inputs with best model from last training run''')
    predict_parser.add_argument('--predict', default='predict', dest='task')
    predict_parser.add_argument('--show-plot', action='store_true', dest='show_plot')
    predict_parser.add_argument('--save-plot', action='store_true',  dest='save_plot')
    args = parser.parse_args()
    print(args)
    if args.log_filter:
        log.addFilter(CMDFilter(args.log_filter))

    ROOT_DIR = initialize_task(args.hpconfig, prefix=args.results_dir)

    sys.path.append('.')
    print(sys.path)

    args.hpconfig = args.hpconfig.replace('/', '.').replace('.py', '')
    modpath = args.hpconfig.split('.')
    pkg, mod = '.'.join(modpath[:-1]), modpath[-1]
    
    HPCONFIG = importlib.import_module(args.hpconfig)
    config.HPCONFIG = HPCONFIG.CONFIG
    config.ROOT_DIR = ROOT_DIR
    config.NAME = SELF_NAME
    print('====================================')
    print(ROOT_DIR)
    print('====================================')

    load_data = getattr(utilz, config.HPCONFIG.dataset_name, 'load_data')
    #cache_filepath = '{}/{}__{}__cache.pkl'.format(ROOT_DIR, config.HPCONFIG.dataset_name, SELF_NAME)
    cache_filepath = '{}__{}__cache.pkl'.format(config.HPCONFIG.dataset_name, SELF_NAME)
    if config.CONFIG.flush:
        log.info('flushing...')
        dataset = load_data(config, max_sample_size=config.HPCONFIG.max_samples)
        pickle.dump(dataset, open(cache_filepath, 'wb'))
    else:
        try:
            dataset = pickle.load(open(cache_filepath, 'rb'))
        except:
            dataset = load_data(config, max_sample_size=config.HPCONFIG.max_samples)
            pickle.dump(dataset, open(cache_filepath, 'wb'))
            
    log.info('dataset train size: {}'.format(len(dataset.trainset)))
    log.info('dataset test size: {}'.format(len(dataset.testset)))
    log.info('dataset outvocab size: {}'.format(len(dataset.output_vocab)))
    log.info('dataset outvocab: {}'.format(dataset.output_vocab.word2index))
    
    for i in range(2):
        log.info('random sample: {}'.format(
            pformat(random.choice(dataset.trainset))))

    #log.info('vocab: {}'.format(pformat(dataset.output_vocab.freq_dict)))
    ##############################################################
    # model snapshot data 
    ##############################################################

    if config.HPCONFIG.model_name == 'CharacterModel':
        net =  model.CharacterModel(config, 'CharacterModel',
                           dataset = dataset,
        )
        
    print('**** the model', net)
    net.restore_checkpoint()
    
    if config.CONFIG.cuda:
        net = net.cuda()        
        if config.CONFIG.multi_gpu and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)
    
    if args.task == 'train':
        net.do_train()

    if args.task == 'drop-words-and-validate':
        net.drop_words_and_validate(args.epoch)
        
    if args.task == 'dump-vocab':
        from collections import Counter
        from utilz import Sample
        counter = Counter()
        for s in dataset.trainset:
            counter.update([s.word, s.context])

        embedding = []
        words = sorted(counter.keys())
        for w in tqdm(words):
            ids, word, context = _batchop([Sample('0', w, '')], for_prediction=True)
            emb = net.__(net.embed(word), 'emb')
            embedding.append(emb)

        embedding = torch.stack(embedding).squeeze()
        dump_vocab_tsv(config,
                       words,
                       embedding.cpu().detach().numpy(),
                       config.ROOT_DIR + '/vocab.tsv')

        
    if args.task == 'dump-cosine-similarity':
        dump_cosine_similarity_tsv(config,
                   dataset.input_vocab,
                   net.embed.weight.data.cpu(),
                   config.ROOT_DIR + '/cosine_similarity.tsv')
        
        
    if args.task == 'predict':
        for i in range(10):
            try:
                output = net.do_predict(id=i)
            except:
                log.exception('######### Predict ###########')
                pass
                
