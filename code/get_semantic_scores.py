import numpy as np
from pprint import pprint
from data_prep import get_depr_lexicon_data, get_data, get_vocab
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def get_embedding(fname):
    '''
    returns a dict of word embeddings given file name.
    the file must have line_i as "word_i d1 d2 ... d300"
    '''
    embedding = {}
    with open(fname, 'r') as f:
        for line in f:
            lineArr = line.split(' ')
            word = lineArr[0]
            vector = lineArr[1:-1]  # exclude the word and \n
            vector = np.array(vector).astype(float)
            embedding[word] = vector
    return embedding


def get_lexicon_embedding_matrix(lexicon, embedding, DIM):
    '''
    get a LEXICON_SIZE X EMBEDDING_DIM matrix, one row for each word's embedding
    '''
    lexicon_embedding = {}
    for category in lexicon.values():
        for phrase in category:
            words = phrase.split(' ')
            n = len(words)
            # only consider unigrams
            if n > 1:
                continue
            if phrase in embedding:
                lexicon_embedding[phrase] = embedding[phrase]
    embedding_matrix = np.zeros((len(lexicon_embedding), DIM))
    for index, vector in enumerate(lexicon_embedding.values()):
        embedding_matrix[index] = vector
    return np.matrix(embedding_matrix)


def main():
    vocab = get_vocab()
    embedding = get_embedding(
        '../models/retrofitted_cnet_emb_file_step_10.txt')
    VEC_DIM = 300
    lexicon = get_depr_lexicon_data(concat=False)
    lexicon_embedding_matrix = get_lexicon_embedding_matrix(
        lexicon, embedding, VEC_DIM)
    print(f'SHAPE OF LEX_EMB_MATRIX = {lexicon_embedding_matrix.shape}')
    semantic_scores = {}
    for word in vocab:
        if any([word in lexicon_category for lexicon_category in lexicon.values()]):
            semantic_scores[word] = 1
        else:
            if word in embedding:
                vector = embedding[word]
            else:
                vector = np.zeros(VEC_DIM)
            vector = np.matrix(vector.reshape((VEC_DIM, 1)))
            semantic_scores[word] = (lexicon_embedding_matrix * vector).max()
    with open('../other_data/semantic_scores.txt', 'w') as f:
        lines = [f'{word}\t{score}' for word, score in semantic_scores.items()]
        f.write('\n'.join(lines))

    scores = np.array(list(semantic_scores.values())).reshape(-1, 1)
    sns.distplot(scores, kde=False)
    plt.show()
    # scores = StandardScaler().fit_transform(scores) + 1
    # sns.distplot(scores, kde=False)
    # plt.show()


if __name__ == '__main__':
    main()
