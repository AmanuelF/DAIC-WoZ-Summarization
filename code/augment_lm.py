from time import time
import pandas as pd
from data_prep import get_depr_lexicon_data, get_sem_scores
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns


def constrain_lm_logprob(logprob):
    if logprob < -99:
        return -99
    elif logprob > 0:
        return -0.01
    else:
        return logprob


def augment_simple():
    '''
    Old augmentation technique, setting lexicon unigrams' logprob to -1.
    Creates a new augmented language model and saves the file.
    '''
    data = get_depr_lexicon_data(concat=False)
    lexiconList = pd.Series([s for s in data.values()]).sum()
    lexiconSet = set(lexiconList)
    to_remove = set()

    # ensure only unigrams are present
    for phrase in lexiconList:
        if ' ' in phrase:
            to_remove.add(phrase)
    print(f'lexicon length = {len(lexiconSet)}')
    lexiconSet -= to_remove
    print(f'number of unigrams = {len(lexiconSet)}')
    lines = []
    with open('../models/en-70k-0.2.lm', 'r') as fp:
        lines = fp.readlines()
        print('DONE READING')
    tempArr = lines[2].strip().split('=')
    num_1_grams_old = int(tempArr[-1])

    # for each unigram, set its log prob to -1 if it's in lexicon
    for i in range(7, num_1_grams_old + 7):
        lineArr = lines[i].split('\t')
        if lineArr[1] in lexiconSet:
            # print(f'found {lineArr[1]}')
            lexiconSet.remove(lineArr[1])
            lineArr[0], lineArr[2] = '-1.000000', '-0.999000\n'
            lines[i] = '\t'.join(lineArr)
        elif len(lineArr) == 2 and lineArr[1][:-1] in lexiconSet:
            # if the unigram doesn't have the second logprob (see ARPA language model format)
            print(f'found {lineArr[1][:-1]} with \\n')
            lexiconSet.remove(lineArr[1][:-1])
            lineArr[0] = '-1.000000\n'
            lines[i] = '\t'.join(lineArr)
    # words in lexicon but not in LM
    print(f'not in orig model: {len(lexiconSet)}')
    n_1_grams_new = num_1_grams_old + len(lexiconSet)
    tempArr[-1] = f'{n_1_grams_new}\n'
    lines[2] = '='.join(tempArr)
    for word in lexiconSet:
        lines.insert(num_1_grams_old+7, f'-1.000000\t{word}\t-0.999000\n')
    with open('../models/en-70k-0.2_AUGMENTED.lm', 'w') as fp:
        fp.write(''.join(lines))


def augment_sem_scores():
    '''
    Language model augmentation by using cosine score between lexicon terms and terms in language model.
    Creates a new augmented language model and saves the file.
    '''
    scores = get_sem_scores(normalize=True, log=True, scalingfactor=2)
    # avg = sum(scores.values()) / len(scores)
    # print(avg)
    # sns.distplot(list(scores.values()), kde=False)
    # plt.show()
    lines = []
    with open('../models/en-70k-0.2_PRUNED.lm', 'r') as fp:
        lines = fp.readlines()
        print('DONE READING')
    tempArr = lines[2].strip().split('=')
    num_1_grams_old = int(tempArr[-1])
    tempArr = lines[3].strip().split('=')
    num_2_grams_old = int(tempArr[-1])
    tempArr = lines[4].strip().split('=')
    num_3_grams_old = int(tempArr[-1])

    t1 = time()

    new_lm = lines[:7]
    num_1_grams_new = 0
    for i in range(7, num_1_grams_old + 7):
        lineArr = lines[i].strip().split('\t')
        logprob = float(lineArr[0])
        word = lineArr[1]
        score = scores[word]

        # adding to logprob is same as multiplying with prob
        lineArr[0] = str(constrain_lm_logprob(logprob + score))
        if len(lineArr) > 2:
            logprob_other = float(lineArr[2].strip())
            lineArr[2] = str(constrain_lm_logprob(logprob_other + score))
        new_lm.append('\t'.join(lineArr) + '\n')
        num_1_grams_new += 1
    new_lm.append('\n\\2-grams:\n')

    t2 = time()
    print(f'TIME FOR 1-GRAMS: {int(t2-t1)}')

    num_2_grams_new = 0
    for i in range(num_1_grams_old + 9, num_1_grams_old + num_2_grams_old + 9):
        lineArr = lines[i].split('\t')
        logprob = float(lineArr[0])
        word1, word2 = lineArr[1].strip().split(' ')

        # score of n-gram is max score of its constituent words
        score = max((scores[word1], scores[word2]))
        lineArr[0] = str(constrain_lm_logprob(logprob + score))
        if len(lineArr) > 2:
            logprob_other = float(lineArr[2].strip())
            lineArr[2] = str(constrain_lm_logprob(
                logprob_other + score)) + '\n'
        new_lm.append('\t'.join(lineArr))
        num_2_grams_new += 1
    new_lm.append('\n\\3-grams:\n')

    t3 = time()
    print(f'TIME FOR 2-GRAMS: {int(t3-t2)}')

    num_3_grams_new = 0
    for i in range(num_1_grams_old + num_2_grams_old + 11, num_2_grams_old + num_3_grams_old + 11):
        lineArr = lines[i].split('\t')
        logprob = float(lineArr[0])
        word1, word2, word3 = lineArr[1].strip().split(' ')

        # score of n-gram is max score of its constituent words
        score = max((scores[word1], scores[word2], scores[word3]))
        lineArr[0] = str(constrain_lm_logprob(logprob + score))
        if len(lineArr) > 2:
            logprob_other = float(lineArr[2].strip())
            lineArr[2] = str(constrain_lm_logprob(
                logprob_other + score)) + '\n'
        new_lm.append('\t'.join(lineArr))
        num_3_grams_new += 1
    new_lm.append('\n\\end\\\n')

    t4 = time()
    print(f'TIME FOR 3-GRAMS: {int(t4-t3)}')

    new_lm[2] = f'ngram 1={num_1_grams_new}\n'
    new_lm[3] = f'ngram 2={num_2_grams_new}\n'
    new_lm[4] = f'ngram 3={num_3_grams_new}\n'

    with open('../models/en-70k-0.2_PRUNED_AUGMENTED_SEM.lm', 'w') as fp:
        fp.write(''.join(new_lm))


def main():
    # augment_simple()
    augment_sem_scores()


if __name__ == '__main__':
    main()
