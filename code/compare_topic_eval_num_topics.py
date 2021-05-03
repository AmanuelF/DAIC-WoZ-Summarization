from json import load
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_prep import to_exclude


def main_old():
    '''
    Script to compare median scores of different summarization algorithms for dirrent number of topics.
    Saves line plot image files.
    '''
    medians = defaultdict(list)
    algos = ('SEM_AUG', 'EXT', 'NORMAL', 'OVER_EXT')
    for n_topics in range(2, 9):
        lim = 7
        metric = 'KL'
        '''
        dictUnAug, dictAug, dictExt, dictAbsOnExt = {}, {}, {}, {}
        extractive_algo = 'SumBasic_unpruned'
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

        try:
            os.mkdir(f'../evaluation/n_topics_{n_topics}/images')
        except:
            pass

        auglabel = 'KIAS'
        unauglabel = 'AS'
        extlabel = 'ES'
        absOnExtLabel = 'AOES'
        medians[auglabel].append(np.median(valsAug))
        medians[unauglabel].append(np.median(valsUnAug))
        medians[extlabel].append(np.median(valsExt))
        medians[absOnExtLabel].append(np.median(valsAbsOnExt))

    for key, vals in medians.items():
        sns.lineplot(x=range(2, 9), y=vals, label=key, ci=ci, err_style='band')
    '''
    for algo in algos:
        data = pd.read_csv(f'../evaluation/{algo}.tsv', sep='\t')
        sns.lineplot(x='num_topics', y='KL', ci='sd', label=algo, data=data)
    plt.legend()
    plt.xlabel('Number of topics used for evaluation')
    plt.ylabel('Median of KL divergences of patients')
    plt.savefig('../evaluation/num_topics_vs_KL_median_new.jpg')


def main():
    algos = ('KIAS', 'ES', 'AS', 'AOES')
    for algo in algos:
        print(f'**********{algo}')
        df = pd.read_csv(f'../evaluation/{algo}.tsv', sep='\t')
        for num_topics in range(2,9):
            ind = df['num_topics'] == num_topics
            values = df[ind]['KL']
            median = round(values.median(), 2)
            std = round(values.std(), 2)
            print(f'n = {num_topics}; median = {median}; std = {std}')


if __name__ == '__main__':
    main()
