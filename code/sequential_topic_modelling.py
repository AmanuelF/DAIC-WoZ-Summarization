import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaSeqModel
from data_prep import get_data
from pprint import pprint


def split(a, n):
    # https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def get_ldaseq_model(data, doc_type, n_bins):
    binned_docs = [[] for i in range(n_bins)]
    for pData in data.values():
        if doc_type == 'question':
            docs = pData.loc[pData['speaker'] == 'ellie', 'value'].tolist()
        elif doc_type == 'answer':
            docs = pData.loc[pData['speaker'] ==
                             'participant', 'value'].tolist()
        elif doc_type == 'both':
            docs = pData['value'].tolist()
        else:
            raise Exception('doc_type must be "question", "answer", or "both"')
        try:
            docs.remove('')
        except ValueError:
            pass
        docs = list(split(docs, n_bins))
        for bin_ind in range(n_bins):
            # binned_docs[bin_ind].append(docs[bin_ind])
            binned_docs[bin_ind] += docs[bin_ind]
    allDocs = pd.Series(binned_docs).sum()
    allDocs = [doc.split(' ') for doc in allDocs]
    dct = Dictionary(allDocs)
    corpus = [dct.doc2bow(row) for row in allDocs]
    time_slice = [len(doc_bin) for doc_bin in binned_docs]
    assert sum(time_slice) == len(allDocs)
    print('Training LdaSeqModel...')
    model = LdaSeqModel(corpus=corpus, id2word=dct,
                        time_slice=time_slice, num_topics=5)
    return model


def save_lda_model(n_bins, docType):
    data, _labels = get_data(preprocess=True, remove_stopwords=True, stem=True)
    model = get_ldaseq_model(data, docType, n_bins)
    model.save(f'../models/ldaseqmodel_{docType}_{n_bins}_timesteps.pkl')


def main():
    docType = 'question'
    for n_bins in (5, 10, 20):
        save_lda_model(n_bins, docType)
    '''
    docType = 'question'
    n_bins = 5
    model = LdaSeqModel.load(f'../models/ldaseqmodel_{docType}_{n_bins}_timesteps.pkl')
    topics = model.print_topics(time=2)
    pprint(topics)
    '''


if __name__ == '__main__':
    main()
