import re
from json import load
import pandas as pd
import gensim.parsing.preprocessing as pp
from math import log10


not_present = {342, 394, 398, 460}  # see dataset documentation
imperfect = {373, 444, 451, 458, 480}  # see dataset documentation
to_exclude = imperfect | not_present
useless_words = {'umm', 'um', 'uhh', 'uh', 'mm', 'mhm'}
summariesDir = '../other_repos/AbTextSumm/my_summaries'


def get_sem_scores(normalize=False, log=False, scalingfactor=1):
    '''
    Returns a dict of word: score pairs.
    `normalize` centers the mean of the scores around 1 by multiplicative scaling.
    `log` return the log of the scores base 10.
    `scalingfactor` is what is multiplied by the log scores before being returned.
    '''
    scores = {}
    with open('../other_data/semantic_scores.txt', 'r') as f:
        for line in f:
            word, score = line[:-1].split('\t')
            scores[word] = float(score)
    if normalize:
        avg = sum(scores.values()) / len(scores)
        for word in scores.keys():
            scores[word] *= 1/avg
    if log:
        for word in scores.keys():
            if scores[word] < 0.0001:
                scores[word] = -2
            else:
                scores[word] = log10(scores[word])
            scores[word] *= scalingfactor
    return scores


def get_vocab(data=None):
    '''
    Returns a set of tokens in the vocabulary, give a dict of dataframes.
    If `data` is None or not given, then the vocabulary of the original conversations is returned.
    '''
    if data is None:
        data = get_data()[0]
    vocab = set()
    for interview in data.values():
        tokenized = interview['value'].apply(lambda s: s.split(' '))
        words = tokenized.sum()
        vocab |= set(words)
    vocab.remove('')
    vocab.add('participant')
    vocab.add('<s>')
    vocab.add('</s>')
    return vocab


def get_data(preprocess=True, stem=False, remove_stopwords=False):
    '''
    Returns a tuple of size 2, where the first is conversation data and the second consists of labels.
    The first is dict of pid: conversation_dataframe, the second is a dataframe of albels.
    `preprocess` removes some redundant utterances, fixes column headings and other basic preprocessing.
    `stem` stems text if `preprocess` is True.
    `remove_stopwords` removes stopwords if `preprocess` is True. 
    '''
    allData = {}
    for pid in range(300, 493):
        if pid in to_exclude:
            continue
        allData[pid] = pd.read_csv(
            f'../data/folders/{pid}_P/{pid}_TRANSCRIPT_CONTINUOUS.csv', sep='\t')
        allData[pid]['speaker'] = allData[pid]['speaker'].apply(
            lambda s: s.lower())

    labels = pd.read_csv('../data/folders/labels/all_labels.csv')
    if preprocess:
        for pid, data in allData.items():
            if data.loc[0, 'speaker'] == 'participant':
                data.drop(0, axis=0, inplace=True)
                data.reset_index(drop=True, inplace=True)
            indParticipant = data['speaker'] == 'participant'
            data.loc[indParticipant, 'value'] = data.loc[indParticipant,
                                                         'value'].apply(lambda s: preprocess_str(s, 'participant'))
            indEllie = data['speaker'] == 'ellie'
            data.loc[indEllie, 'value'] = data.loc[indEllie,
                                                   'value'].apply(lambda s: preprocess_str(s, 'ellie'))
            if remove_stopwords:
                data.loc[indParticipant, 'value'] = data.loc[indParticipant,
                                                             'value'].apply(pp.remove_stopwords)
                data.loc[indEllie, 'value'] = data.loc[indEllie,
                                                       'value'].apply(pp.remove_stopwords)
            if stem:
                data.loc[indParticipant, 'value'] = data.loc[indParticipant,
                                                             'value'].apply(pp.stem_text)
                data.loc[indEllie, 'value'] = data.loc[indEllie,
                                                       'value'].apply(pp.stem_text)
    return allData, labels


def get_pruned_data():
    '''
    Returns a tuple of size 2, where the first is pruned conversation data from lexicon and the second consists of labels.
    The first is dict of pid: conversation_dataframe, the second is a dataframe of albels.
    '''
    allData = {}
    for pid in range(300, 493):
        if pid in to_exclude:
            continue
        allData[pid] = pd.read_csv(
            f'../data/pruned/{pid}.csv', sep='\t')
    labels = pd.read_csv('../data/folders/labels/all_labels.csv')
    return allData, labels


def get_phq8_data(preprocess=True, stem=False, remove_stopwords=False):
    '''
    Returns a dataframe of phq8 questions.
    `preprocess` applies some basic processing.
    `stem` stems text.
    `remove_stopwords` removes stopwords. 
    '''
    phq8 = pd.read_csv('../other_data/phq8.tsv', sep='\t')
    if preprocess:
        phq8['query'] = phq8['query'].apply(
            lambda s: preprocess_str(s, 'participant'))
    if remove_stopwords:
        phq8['query'] = phq8['query'].apply(pp.remove_stopwords)
    if stem:
        phq8['query'] = phq8['query'].apply(pp.stem_text)
    return phq8


def get_depr_lexicon_data(preprocess=True, stem=False, remove_stopwords=False, concat=True):
    '''
    if `concat` is True, returns one string of the entire lexicon seperated by spaces.
    else returns a dict with 10 key: value pairs, one for each category in the lexicon.
    `preprocess` applies some basic processing.
    `stem` stems text.
    `remove_stopwords` removes stopwords. 
    '''
    with open('../other_data/depression_lexicon.json', 'r') as f:
        jsonData = load(f)
    if not concat:
        if preprocess:
            for key, val in jsonData.items():
                val = [s.replace('_', ' ') for s in val]
                jsonData[key] = [preprocess_str(s, 'participant') for s in val]
        return jsonData
    else:
        data = pd.DataFrame(columns=['query'])
        data['query'] = pd.Series([' '.join(wordList)
                                   for wordList in jsonData.values()])
        data['query'] = data['query'].apply(lambda s: s.replace('_', ' '))
        if preprocess:
            data['query'] = data['query'].apply(
                lambda s: preprocess_str(s, 'participant'))
        if remove_stopwords:
            data['query'] = data['query'].apply(pp.remove_stopwords)
        if stem:
            data['query'] = data['query'].apply(pp.stem_text)
        return data


def get_summaries(augmented_lm, sem, lim=7, over_ext=False):
    '''
    Returns a dict of pid: abstract_summary.
    `preprocess` applies some basic processing.
    `stem` stems text.
    `remove_stopwords` removes stopwords. 
    '''
    summaries = {}
    if over_ext:
        src = f'{summariesDir}/my_summaries_over_ext'
    elif sem:
        src = f'{summariesDir}/my_summaries_sem'
    else:
        src = f'{summariesDir}/my_summaries'
    if augmented_lm:
        src += f'_augmented_lim_{lim}'
    else:
        src += f'_lim_{lim}'
    for pid in range(300, 493):
        if pid in to_exclude:
            continue
        with open(f'{src}/{pid}.txt', 'r') as f:
            summaries[pid] = f.read().lower()
    return summaries


def get_ext_summaries(algo):
    '''
    Returns a dict of pid: extractive_summary.
    '''
    summaries = {}
    for pid in range(300, 493):
        if pid in to_exclude:
            continue
        src = f'../extractive_summaries/{algo}'
        with open(f'{src}/{pid}.txt', 'r') as f:
            summaries[pid] = f.read().lower()
    return summaries


def decontracted(phrase):
    '''
    converts "n't" to " not", etc.
    '''
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


# preprocess data
def preprocess_str(s, speaker):
    '''
    speaker specific preprocess function.
    same for both, but ellie also has some standard questions inside brackets.
    '''
    res = str(s).lower()
    res = res.replace('_', '')
    for word in useless_words:
        res = res.replace(word, '')
    res = re.sub('<[^>]+>', '', res)  # remove stuff like <laughter>
    res = re.sub('\\[[^>]+\\]', '', res)  # remove stuff like [laughter]
    if speaker == 'ellie' and '(' in s and ')' in s:
        # https://stackoverflow.com/a/14597158
        resList = re.findall(r'.*?\((.*?)\)', res)
        res = ' '.join(resList)
    res = decontracted(res)
    res = pp.strip_tags(res)
    res = pp.strip_punctuation(res)
    res = pp.strip_numeric(res)
    # res = pp.remove_stopwords(res)
    res = pp.strip_multiple_whitespaces(res)
    # res = pp.strip_short(res)
    res = res.lower()
    return res


def to_passage_simple(data, sep='\n'):
    '''
    convert a conversation to a passage with some simple text replacements
    '''
    text = ''
    for _ind, row in data.iterrows():
        if row['speaker'] == 'ellie':
            temp = (' ' + str(row['value'])).replace(' you ',
                                                     ' they ').replace(' i ', ' she ').replace(' am ', ' is ')
            text += f'participant was asked{temp}, '
        elif row['speaker'] == 'participant':
            temp = (' ' + str(row['value'])).replace(' you ',
                                                     ' she ').replace(' i ', ' they ').replace(' am ', ' is ')
            text += f'then {row["speaker"]} said{temp} .{sep}'
    return text


def get_simple_passages():
    '''
    Returns a dict of pid: passage.
    '''
    allPassages = {}
    for pid in range(300, 493):
        if pid in to_exclude:
            continue
        with open(f'../data/simple_passages/{pid}.txt', 'r') as fp:
            allPassages[pid] = fp.read()
    return allPassages
