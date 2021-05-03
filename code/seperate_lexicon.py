from data_prep import get_depr_lexicon_data


def main():
    '''
    script to save each category in depression lexicon its own file.
    required for retrofitting vectors.
    '''
    lexicon = get_depr_lexicon_data(concat=False)
    for key, value in lexicon.items():
        with open(f'../other_data/depression_lexicon_seperated/{key}.txt', 'w') as f:
            f.write('\n'.join(value))


if __name__ == '__main__':
    main()
