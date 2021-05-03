import pandas as pd


def make_continuous(df):
    '''
    given a conversation dataframe where one speaker may have multiple consecutive utterances,
    append consecutive utterances so that each speaker has one line at a time.
    rerturns the continuous dataframe of the conversation.
    '''
    new_df = pd.DataFrame(columns=df.columns)
    for index, row in df.iterrows():
        if index == 0:
            new_df = new_df.append(row)
        elif row['speaker'] == new_df.loc[new_df.index[-1], 'speaker']:
            new_df.loc[new_df.index[-1], 'value'] = str(
                new_df.loc[new_df.index[-1], 'value']) + ' ' + str(row['value'])
        else:
            new_df = new_df.append(row)
        new_df.loc[new_df.index[-1], 'stop_time'] = row['stop_time']
    return new_df


def main():
    not_present = (342, 394, 398, 460)
    to_exclude = not_present + (373, 444, 451, 458, 480)
    for pid in range(300, 493):
        if pid in to_exclude:
            continue
        df = pd.read_csv(
            f'../data/folders/{pid}_P/{pid}_TRANSCRIPT.csv', sep='\t', dtype={'value': object})
        cont_df = make_continuous(df)
        cont_df.to_csv(
            f'../data/folders/{pid}_P/{pid}_TRANSCRIPT_CONTINUOUS.csv', sep='\t', index=False)


if __name__ == '__main__':
    main()
