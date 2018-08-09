import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def main():
    use_brown = True
    n_files = 50

    if use_brown:
        we_file = 'glove_model_brown.npz'
        w2i_file = 'glove_word2idx_brown.json'
    else:
        we_file = 'glove_model_%s.npz' % n_files
        w2i_file = 'glove_word2idx_%s.json' % n_files

    words = ['japan', 'japanese', 'england', 'english', 'australia', 
             'australian', 'china', 'chinese', 'italy', 'italian', 
             'french', 'france']

    with open(w2i_file) as f:
        word2idx = json.load(f)

    npz = np.load(we_file)
    W = npz['arr_0']
    U = npz['arr_1']
    We = (W + U.T) / 2

    idx = [word2idx[w] for w in words]

    tsne = TSNE()
    Z = tsne.fit_transform(We)
    Z = Z[idx]

    plt.scatter(Z[:,0], Z[:,1])
    for i in range(len(words)):
        plt.annotate(s=words[i], xy=(Z[i,0], Z[i,1]))
    plt.show()


if __name__ == '__main__':
    main()

