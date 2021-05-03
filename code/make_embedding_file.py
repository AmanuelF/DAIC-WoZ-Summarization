import numpy as np
from pickle import load
from data_prep import get_vocab, get_data


def main():
    '''
    script to convert the pickled concept-net embedding into ascii format.
    the saved embedding file has line_i as "word_i d1 d2 ... d300"
    only considers words in the conversation vocabulary.
    '''
    vocab = get_vocab()
    with open('../../embeddings/only_english_cnet_vectors_np.pkl', 'rb') as f:
        embedding_dct = load(f, encoding='latin1')
    print(list(embedding_dct.keys())[0])
    DIM = len(embedding_dct['/c/en/diagnostic'])
    filestr = ''
    for word in sorted(vocab):
        key = f'/c/en/{word}'
        if key in embedding_dct:
            word_vec = embedding_dct[key].astype(str)
        else:
            word_vec = np.zeros(shape=DIM).astype(str)
        filestr += f'{word} {" ".join(word_vec)}\n'
    with open('../models/vocab_cnet_emb_file.txt', 'w') as f:
        f.write(filestr)


if __name__ == '__main__':
    main()
