import os
import re
import sys
sys.path.append('..')
import pickle
import glob

import importlib
from collections import defaultdict

from pprint import pprint, pformat

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.style.use('ggplot')

from anikattu.utilz import initialize_task, pad_seq

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

colors = {
    1: '#DC143C',
    2: '#8B0000',
    3: '#FF1493',
    4: '#DB7093',
    5: '#FF6347',
    6: '#FF4500',
    7: '#FFFF00',
    8: '#FFDAB9',
    9: '#EE82EE',
    10: '#FF00FF',
    11: '#663399',
    12: '#4B0082',
    13: '#ADFF2F',
    14: '#00FF00',
    15: '#2E8B57',
    16: '#808000',
    17: '#008080',
    18: '#00FFFF',
    19: '#00BFFF',
    20: '#2F4F4F',
    21: '#000000',
}

PREFIX = 'hpconfig.amazon'

hpconfigs = ['hpconfig_c', 'hpconfig_w'] #, 'hpconfig_wc', 'hpconfig_kv']

run_dirs = ['run00', 'run01']

EPOCH_COUNT_MIN = 0
EPOCH_COUNT_MAX = 100

def read_pkls(hpconfigs=hpconfigs):
    root_dirs = defaultdict(list)
    accuracies = defaultdict(list)
    losses = defaultdict(list)

    for hpconfig in hpconfigs:
        try:
            hpconfig = PREFIX + '.' + hpconfig
            HPCONFIG = importlib.import_module(hpconfig)
            model = re.search('.*hpconfig_(\w+)', hpconfig)
            model = model.group(1)

            log.info('hpconfig -- {}'.format(hpconfig))
            
            for rd in run_dirs:
                rd = initialize_task(hpconfig.replace('.', '/') + '.py', rd)
                root_dirs[model].append(rd)
                log.info(' rd: {}'.format(rd))

                f = '{}/results/metrics/{}.test_accuracy.pkl'.format(
                    rd
                    , HPCONFIG.CONFIG.model_name
                )
                
                accuracies[model].append(
                    pickle.load(open(f,'rb'))
                )

                
                f = '{}/results/metrics/{}.test_loss.pkl'.format(
                     rd
                    , HPCONFIG.CONFIG.model_name
                )
                
                losses[model].append(
                    pickle.load(open(f,'rb'))
                )
                
            """
            if len(accuracies[model]) < EPOCH_COUNT_MIN:
                EPOCH_COUNT_MIN = len(accuracies[model])
                print('min_epoch_count: {}'.format( EPOCH_COUNT_MIN))

            if len(accuracies[model]) > EPOCH_COUNT_MAX:
                EPOCH_COUNT_MAX = len(accuracies[model])
                print('max_epoch_count: {}'.format(EPOCH_COUNT_MAX))
            """
        except:
            log.exception('{} not found'.format(model))
            
    return root_dirs, accuracies, losses, EPOCH_COUNT_MIN, EPOCH_COUNT_MAX



def calc_moving_avg(p, N = 5):
    return np.convolve(p , np.ones((N,))/N, mode='same')[:-1]

def plot_accuracies(epoch_limit,
                    min_epoch_count, max_epoch_count,
                    accuracies,
                    plot_title='Combined Accuracy',
                    plot_filepath='combined_accuracy_heatmap.png',
                    labels = {},
                    y_offsets = {},
                    ylabel = 'Accuracy',
                    xlabel = 'Epoch',
                    ylim = (0, 1),
                    moving_avg = 0,
):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    #fig.subplots_adjust(left=.06, right=.75, bottom=.02, top=.94)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    for i, (task_name, acc) in enumerate(accuracies):
        p = np.asarray(pad_seq(acc)).mean(axis=0)
        log.debug('p.shape: {}'.format(p.shape))
        if moving_avg:
            p = calc_moving_avg(p, moving_avg)
            
        line = plt.plot(p,
                        lw=2.5,
                        color=colors[i+1],
                        label=task_name)
        plt.legend(loc='lower right')

        # Add a text label to the right end of every line. Most of the code below
        # is adding specific offsets y position because some labels overlapped.
        y_pos = acc[-1] #- 0.5


        # Again, make sure that all labels are large enough to be easily read
        # by the viewer.
        task_name = os.path.basename(task_name)

        if task_name in y_offsets:
            y_pos += y_offsets[task_name]

        if task_name in labels:
            task_name = labels[task_name]

    fig.suptitle(plot_title, fontsize=18, ha='center')

    if ylim:
        plt.ylim(*ylim)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(plot_filepath, bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    root_dirs, accuracies, losses, min_epoch_count, max_epoch_count = read_pkls()

    pprint(accuracies)
    pprint(losses)
    pprint(root_dirs)

    
    losses = losses.items()
    plot_accuracies(EPOCH_COUNT_MAX,
                    min_epoch_count, max_epoch_count,
                    losses,
                    'Loss',
                    'loss.png',
                    ylabel='Loss',
                    ylim=None,
                    moving_avg = 0)


    accuracies = accuracies.items()
    plot_accuracies(EPOCH_COUNT_MAX,
                    min_epoch_count, max_epoch_count,
                    accuracies,
                    'Accuracy',
                    'accuracy.png',
                    ylabel='Accuracy',
                    moving_avg = 0)

