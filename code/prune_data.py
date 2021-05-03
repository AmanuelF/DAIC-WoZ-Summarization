from collections import defaultdict
import numpy as np
import pandas as pd
from pickle import load
from sklearn.feature_extraction.text import TfidfVectorizer
from data_prep import get_data, get_phq8_data, get_depr_lexicon_data
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint


def get_orig_indices(data, indices):
    '''
    given indices relative to only questions/answers (like 0th question, 5th question,...),
    return the indices relative to the entire conversation (0th question -> 0th question and 0th answer)
    '''
    orig_inds = {}
    for pid, pIndices in indices.items():
        pData = data[pid]
        orig_inds[pid] = [2*n for n in pIndices] + [2*n+1 for n in pIndices]
        if orig_inds[pid]:
            maxInd = max(orig_inds[pid])
            if maxInd >= len(pData):
                orig_inds[pid].remove(maxInd)
    return orig_inds


def sentence_embedding(sentence, embedding_index, tfidf, vec_dim):
    '''
    get a sentence embedding by computing weighted average of word embeddings,
    where a word's weight is its tfidf score.
    '''
    # tfidf_ind = defaultdict(int, tfidf.vocabulary_)
    tfidf_ind = tfidf.vocabulary_
    tfidfSentence = tfidf.transform([sentence])
    words = sentence.split(' ')
    embedding = np.zeros(shape=vec_dim)
    for word in words:
        if word not in tfidf_ind:
            score = 0
        else:
            score = tfidfSentence[0, tfidf_ind[word]]
        embedding += score * embedding_index[f'/c/en/{word}']
    rootSqSum = (embedding**2).sum() ** 0.5
    if rootSqSum != 0:
        return embedding / rootSqSum
    else:
        return embedding


def add_window(indices, window, maxInd):
    '''
    given some indices, also include those indices in a window before and after them.
    ex.) if 6 is input and window is (1,2) then output would include 5,6,7,8
    '''
    indices = set(indices)
    to_add = set()
    behindWindow, frontWindow = window
    for ind in indices:
        to_add |= set([ind+(j+1) for j in range(frontWindow)])
        to_add |= set([ind-(j+1) for j in range(behindWindow)])
    to_add &= set(range(maxInd))
    return list(indices | to_add)


def get_indices_matched(query, data, speaker, window=(0, 0)):
    '''
    get the indices of the conversation with matches to lexicon, along with the window
    of utterances to also be included for context.
    '''
    phrases = pd.Series([s for s in query.values()]).sum()
    allIndices = {}
    for pid, pData in data.items():
        speakerInd = pData['speaker'] == speaker
        answers = pData[speakerInd]['value'].reset_index(
            drop=True).apply(lambda s: s.split(' '))
        answers = answers.apply(set)
        '''
        for phrase in phrases:
            for ans in answers:
                if phrase in ans.split(' '):
                    print(phrase)
        '''
        indices = [ind for ind in answers.index if any(
            phrase in answers[ind] for phrase in phrases)]
        allIndices[pid] = add_window(indices, window, len(answers))
    return allIndices


def get_similar_cnet(query, data, speaker, window=(0, 0), top_n=1):
    '''
    get relevant utterances using only phq8 questions and conceptnet embeddings.
    '''
    with open('../../embeddings/only_english_cnet_vectors_np.pkl', 'rb') as fp:
        embedding_index = load(fp, encoding='latin1')
    vec_dim = embedding_index['/c/en/boy'].shape
    embedding_index = defaultdict(
        lambda: np.zeros(shape=vec_dim), embedding_index)
    tfidf = get_tfidf(query, data, speaker)
    query['vec'] = query['query'].apply(
        lambda s: sentence_embedding(s, embedding_index, tfidf, vec_dim))
    utteranceEmbeddings = {}
    for pid, pData in data.items():
        ind = pData['speaker'] == speaker
        utteranceEmbeddings[pid] = pData.loc[ind, 'value'].apply(
            lambda s: sentence_embedding(s, embedding_index, tfidf, vec_dim))
    queryEmbeddings = np.array(query['vec'].tolist())
    sim_matrices = {}
    allIndices = {}
    for pid, embeddings in utteranceEmbeddings.items():
        utteranceEmbeddings[pid] = np.array(embeddings.tolist())
        sim_matrices[pid] = np.matmul(
            queryEmbeddings, utteranceEmbeddings[pid].T)
        indices = np.argsort(sim_matrices[pid], axis=1)[:, :top_n].flatten()
        allIndices[pid] = add_window(
            indices, window, len(utteranceEmbeddings[pid]))
    return allIndices


def get_tfidf(query, data, speaker):
    '''
    get a tfidf model of tfidf model of a query.
    '''
    tfidf = TfidfVectorizer(stop_words='english')
    docs = query['query']
    for pData in data.values():
        ind = pData['speaker'] == speaker
        docs = docs.append(pData.loc[ind, 'value'])
    tfidf.fit(docs)
    return tfidf


def get_pruned_indices():
    '''
    get the final pruned indices of conversations.
    uses lexical matching and context window.
    '''
    data, _labels = get_data(
        preprocess=True, remove_stopwords=True, stem=False)

    # questionQuery = get_depr_lexicon_data(preprocess=True, remove_stopwords=True, stem=False, concat=True)
    # question_inds = get_similar_cnet(questionQuery, data, 'ellie', (1,1), 3)
    questionQuery = get_depr_lexicon_data(concat=False)
    question_inds = get_indices_matched(questionQuery, data, 'ellie', (2, 2))
    true_inds_q = get_orig_indices(data, question_inds)

    answerQuery = get_depr_lexicon_data(concat=False)
    answer_inds = get_indices_matched(answerQuery, data, 'participant', (2, 2))
    true_inds_a = get_orig_indices(data, answer_inds)
    # sns.distplot([len(a) for a in true_inds_a.values()])
    # sns.distplot([len(a)/2 for a in data.values()])
    # plt.show()
    # return

    final_inds = {}
    for pid, qInds, aInds in zip(true_inds_q.keys(), true_inds_q.values(), true_inds_a.values()):
        final_inds[pid] = sorted(set(qInds) | set(aInds))

    sns.distplot([len(a) for a in final_inds.values()], color='orange')
    sns.distplot([len(a) for a in data.values()], color='blue')
    plt.show()
    return final_inds


def main():
    '''
    script to prune the conversations and save them.
    '''
    inds = get_pruned_indices()
    # inds = defaultdict(lambda: [0,1,2,3])
    allData, _ = get_data()
    for pid in allData.keys():
        allData[pid] = allData[pid].loc[inds[pid], :]
        allData[pid].to_csv(f'../data/pruned/{pid}.csv', index=False, sep='\t')


if __name__ == '__main__':
    main()
