import numpy as np
import pandas as pd
from data_prep import get_data, get_vocab
from gensim.parsing.preprocessing import strip_punctuation
from pprint import pprint
from time import time


def main():
    '''
    script to remove out-of-vocabulary words from an arpa language model
    '''
    vocab = get_vocab()
    print(f'{len(vocab)} words in vocabulary')
    # pprint(vocab)
    with open('../models/en-70k-0.2.lm', 'r') as fp:
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
        word = lineArr[1]
        if word in vocab:
            new_lm.append('\t'.join(lineArr) + '\n')
            num_1_grams_new += 1
    new_lm.append('\n\\2-grams:\n')

    t2 = time()
    print(f'TIME FOR 1-GRAMS: {int(t2-t1)}')

    num_2_grams_new = 0
    for i in range(num_1_grams_old + 9, num_1_grams_old + num_2_grams_old + 9):
        lineArr = lines[i].split('\t')
        word1, word2 = lineArr[1].strip().split(' ')
        if word1 in vocab and word2 in vocab:
            new_lm.append('\t'.join(lineArr))
            num_2_grams_new += 1
    new_lm.append('\n\\3-grams:\n')

    t3 = time()
    print(f'TIME FOR 2-GRAMS: {int(t3-t2)}')

    num_3_grams_new = 0
    for i in range(num_1_grams_old + num_2_grams_old + 11, num_2_grams_old + num_3_grams_old + 11):
        lineArr = lines[i].split('\t')
        word1, word2, word3 = lineArr[1].strip().split(' ')
        if word1 in vocab and word2 in vocab and word3 in vocab:
            new_lm.append('\t'.join(lineArr))
            num_3_grams_new += 1
    new_lm.append('\n\\end\\\n')

    t4 = time()
    print(f'TIME FOR 3-GRAMS: {int(t4-t3)}')

    new_lm[2] = f'ngram 1={num_1_grams_new}\n'
    new_lm[3] = f'ngram 2={num_2_grams_new}\n'
    new_lm[4] = f'ngram 3={num_3_grams_new}\n'

    with open('../models/en-70k-0.2_PRUNED.lm', 'w') as fp:
        fp.write(''.join(new_lm))

    # pprint(new_lm)
    print(len(new_lm))


if __name__ == '__main__':
    main()
