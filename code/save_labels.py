import pandas as pd


def main():
    '''
    save all the train, test and dev labels to one csv
    '''
    train_labels = pd.read_csv(
        '../data/folders/labels/train_split_Depression_AVEC2017.csv', usecols=(0, 1, 2))
    dev_labels = pd.read_csv(
        '../data/folders/labels/dev_split_Depression_AVEC2017.csv', usecols=(0, 1, 2))
    test_labels = pd.read_csv(
        '../data/folders/labels/full_test_split.csv', usecols=(0, 1, 2))
    test_labels.columns = ['Participant_ID', 'PHQ8_Binary', 'PHQ8_Score']
    all_labels = train_labels.append(dev_labels).append(test_labels)
    all_labels.sort_values(by=('Participant_ID'), inplace=True)
    all_labels.to_csv('../data/folders/labels/all_labels.csv', index=False)


if __name__ == '__main__':
    main()
