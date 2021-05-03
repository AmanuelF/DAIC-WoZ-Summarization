from json import load
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_prep import to_exclude


def main():
    '''
    Script to compare different summarization algorithms for one topic model, for all patients.
    Saves stem plot image files.
    '''
    lim = 7
    metric = 'KL'
    n_topics = 5
    dictUnAug, dictAug, dictExt, dictAbsOnExt = {}, {}, {}, {}
    extractive_algo = 'SumBasic_unpruned'
    print(f'*****{metric}*****')
    for pid in range(300, 493):
        if pid in to_exclude:
            continue
        with open(f'../evaluation/n_topics_{n_topics}/evaluation_lim_{lim}/{pid}.json', 'r') as fp:
            unaugmented_res = load(fp)
            dictUnAug[pid] = unaugmented_res[metric]
        with open(f'../evaluation/n_topics_{n_topics}/evaluation_lim_{lim}_over_ext/{pid}.json', 'r') as fp:
            abs_on_ext_res = load(fp)
            dictAbsOnExt[pid] = abs_on_ext_res[metric]
        with open(f'../evaluation/n_topics_{n_topics}/evaluation_sem_augmented_lim_{lim}/{pid}.json', 'r') as fp:
            augmented_res = load(fp)
            dictAug[pid] = augmented_res[metric]
        with open(f'../evaluation/n_topics_{n_topics}/evaluation_{extractive_algo}/{pid}.json', 'r') as fp:
            ext_res = load(fp)
            dictExt[pid] = ext_res[metric]
    valsUnAug, valsAug = np.array(
        list(dictUnAug.values())), np.array(list(dictAug.values()))
    valsExt, valsAbsOnExt = np.array(
        list(dictExt.values())), np.array(list(dictAbsOnExt.values()))
    print(f'median unaugmented: {round(np.median(valsUnAug), 2)}')
    print(f'median augmented  : {round(np.median(valsAug), 2)}')
    print(f'median extractive : {round(np.median(valsExt), 2)}')
    print(f'median abs on ext : {round(np.median(valsAbsOnExt), 2)}')

    pids = sorted(set(range(300, 493)) - to_exclude)
    pids = np.array(pids)
    try:
        os.mkdir(f'../evaluation/n_topics_{n_topics}/images')
    except:
        pass

    plt.figure(figsize=(32, 6))
    plt.tight_layout()
    ax = plt.gca()
    ax.spines['bottom'].set_color('black')

    auglabel = 'PHQxAS'
    unauglabel = 'AS'
    extlabel = 'ES'
    absOnExtLabel = 'AOES'
    xlabel = 'Patient ID'

    plt.stem(pids, valsAug, 'b', markerfmt='bo', label=auglabel)
    plt.stem(pids, valsUnAug, 'm', markerfmt='mo', label=unauglabel)
    plt.stem(pids, valsExt, 'g', markerfmt='go', label=extlabel)
    plt.stem(pids, valsAbsOnExt, 'y', markerfmt='yo', label=absOnExtLabel)
    plt.xlabel(xlabel)
    plt.ylabel(metric)
    plt.legend()
    plt.savefig(f'../evaluation/n_topics_{n_topics}/images/stem_all.png')
    plt.clf()

    improved_idx = np.argwhere(np.logical_and.reduce(
        (valsAug <= valsUnAug, valsAug <= valsExt, valsAug <= valsAbsOnExt)))
    improved_idx = improved_idx.flatten()
    unimproved_idx = np.delete(np.arange(len(pids)), improved_idx)
    print(f'NUM IMPROVED: {len(improved_idx)}')

    plt.stem(pids[improved_idx], valsAug[improved_idx],
             'b', markerfmt='bo', label=auglabel)
    plt.xlim(300, 492)
    plt.ylim(0, 4)
    plt.xticks(pids, rotation='vertical')
    ax = plt.gca()
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=16)
    xticks = ax.xaxis.get_major_ticks()
    for i in unimproved_idx:
        xticks[i].label1.set_visible(False)
    plt.xlabel(xlabel, fontsize=25)
    plt.ylabel(metric, fontsize=25)
    plt.legend(fontsize=25)
    plt.tight_layout()
    plt.savefig(f'../evaluation/n_topics_{n_topics}/images/stem_improved_PHQxAS.png')
    plt.clf()

    plt.stem(pids[improved_idx], valsUnAug[improved_idx],
             'm', markerfmt='mo', label=unauglabel)
    plt.xlim(300, 492)
    plt.ylim(0, 4)
    plt.xticks(pids, rotation='vertical')
    ax = plt.gca()
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=16)
    xticks = ax.xaxis.get_major_ticks()
    for i in unimproved_idx:
        xticks[i].label1.set_visible(False)
    plt.xlabel(xlabel, fontsize=25)
    plt.ylabel(metric, fontsize=25)
    plt.legend(fontsize=25)
    plt.savefig(f'../evaluation/n_topics_{n_topics}/images/stem_improved_as.png')
    plt.clf()

    plt.stem(pids[improved_idx], valsExt[improved_idx],
             'g', markerfmt='go', label=extlabel)
    plt.xlim(300, 492)
    plt.ylim(0, 4)
    plt.xticks(pids, rotation='vertical')
    ax = plt.gca()
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=16)
    xticks = ax.xaxis.get_major_ticks()
    for i in unimproved_idx:
        xticks[i].label1.set_visible(False)
    plt.xlabel(xlabel, fontsize=25)
    plt.ylabel(metric, fontsize=25)
    plt.legend(fontsize=25)
    plt.tight_layout()
    plt.savefig(f'../evaluation/n_topics_{n_topics}/images/stem_improved_ext.png')
    plt.clf()

    plt.stem(pids[improved_idx], valsAbsOnExt[improved_idx],
             'y', markerfmt='yo', label=absOnExtLabel)
    plt.xlim(300, 492)
    plt.ylim(0, 4)
    plt.xticks(pids, rotation='vertical')
    ax = plt.gca()
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=16)
    xticks = ax.xaxis.get_major_ticks()
    for i in unimproved_idx:
        xticks[i].label1.set_visible(False)
    plt.xlabel(xlabel, fontsize=25)
    plt.ylabel(metric, fontsize=25)
    plt.legend(fontsize=25)
    plt.tight_layout()
    plt.savefig(f'../evaluation/n_topics_{n_topics}/images/stem_improved_aoes.png')
    plt.clf()

    plt.stem(pids[unimproved_idx], valsAug[unimproved_idx],
             'b', markerfmt='bo', label=auglabel)
    plt.xlim(300, 492)
    plt.ylim(0, 4)
    plt.xticks(pids, rotation='vertical')
    ax = plt.gca()
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=16)
    xticks = ax.xaxis.get_major_ticks()
    for i in improved_idx:
        xticks[i].label1.set_visible(False)
    plt.xlabel(xlabel, fontsize=25)
    plt.ylabel(metric, fontsize=25)
    plt.legend(fontsize=25)
    plt.tight_layout()
    plt.savefig(f'../evaluation/n_topics_{n_topics}/images/stem_unimproved_PHQxAS.png')
    plt.clf()

    plt.stem(pids[unimproved_idx], valsUnAug[unimproved_idx],
             'm', markerfmt='mo', label=unauglabel)
    plt.xlim(300, 492)
    plt.ylim(0, 4)
    plt.xticks(pids, rotation='vertical')
    ax = plt.gca()
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=16)
    xticks = ax.xaxis.get_major_ticks()
    for i in improved_idx:
        xticks[i].label1.set_visible(False)
    plt.xlabel(xlabel, fontsize=25)
    plt.ylabel(metric, fontsize=25)
    plt.legend(fontsize=25)
    plt.tight_layout()
    plt.savefig(f'../evaluation/n_topics_{n_topics}/images/stem_unimproved_as.png')
    plt.clf()

    plt.stem(pids[unimproved_idx], valsExt[unimproved_idx],
             'g', markerfmt='go', label=extlabel)
    plt.xlim(300, 492)
    plt.ylim(0, 4)
    plt.xticks(pids, rotation='vertical')
    ax = plt.gca()
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=16)
    xticks = ax.xaxis.get_major_ticks()
    for i in improved_idx:
        xticks[i].label1.set_visible(False)
    plt.xlabel(xlabel, fontsize=25)
    plt.ylabel(metric, fontsize=25)
    plt.legend(fontsize=25)
    plt.tight_layout()
    plt.savefig(f'../evaluation/n_topics_{n_topics}/images/stem_unimproved_ext.png')
    plt.clf()

    plt.stem(pids[unimproved_idx], valsAbsOnExt[unimproved_idx],
             'y', markerfmt='yo', label=absOnExtLabel)
    plt.xlim(300, 492)
    plt.ylim(0, 4)
    plt.xticks(pids, rotation='vertical')
    ax = plt.gca()
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=16)
    xticks = ax.xaxis.get_major_ticks()
    for i in improved_idx:
        xticks[i].label1.set_visible(False)
    plt.xlabel(xlabel, fontsize=25)
    plt.ylabel(metric, fontsize=25)
    plt.legend(fontsize=25)
    plt.tight_layout()
    plt.savefig(f'../evaluation/n_topics_{n_topics}/images/stem_unimproved_aoes.png')
    plt.clf()


if __name__ == '__main__':
    main()
