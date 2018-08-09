# Bigram model with Markov assumption

import numpy as np

from util import get_sentences_with_word2idx
from util import get_sentences_with_word2idx_limit_vocab


def get_bigram_probs(sentences, V, start_idx, end_idx, smoothing=1):
    # structure of bigram probability matrix will be:
    # (last word, current word) -> probability
    # we will use add-one smoothing
    # note: we will always ignore this from the END token
    bigram_probs = np.ones((V, V)) * smoothing
    for sentence in sentences:
        for i in range(len(sentence)):

            if i == 0:
                # beginning word
                bigram_probs[start_idx, sentence[i]] += 1
            else:
                # middle word
                bigram_probs[sentence[i - 1], sentence[i]] += 1

            # if we are at the final word
            # we update the bigram for last -> current
            # AND current -> END token
            if i == len(sentence) - 1:
                # final word
                bigram_probs[sentence[i], end_idx] += 1

    # normalize the counts along the rows to get probabilities
    bigram_probs /= bigram_probs.sum(axis=1, keepdims=True)
    return bigram_probs


# a function to calculate normalized log prob score for a sentence
def get_score(sentence, start_idx, end_idx, bigram_probs):
    score = 0
    for i in range(len(sentence)):
        if i == 0:
            # beginning word
            score += np.log(bigram_probs[start_idx, sentence[i]])
        else:
            score += np.log(bigram_probs[sentence[i - 1], sentence[i]])
    # final word
    score += np.log(bigram_probs[sentence[-1], end_idx])

    # normalize the score
    score /= (len(sentence) + 1)

    return score


# a function to map word idx back to real word
def get_words(sentence, idx2word):
    return ' '.join(idx2word[i] for i in sentence)


def main():
    # load the data
    # NOTE: sentences are already converted to sequences of word indexes
    # NOTE: you can limit the vocab size if you run out of memory
    # sentences, word2idx = get_sentences_with_word2idx()
    sentences, word2idx = get_sentences_with_word2idx_limit_vocab(10000)
    idx2word = dict((v, k) for k, v in word2idx.items())

    # vocab size
    V = len(word2idx)
    print('\nVocab size: %d\n' % V)

    # we will also treat beginning of sentence and end of sentence as bigrams
    # START -> first word
    # last word -> END
    start_idx = word2idx['START']
    end_idx = word2idx['END']

    # a matrix where:
    # row = prev word
    # col = curr word
    # value at [row, col] = p(current word | previous word)
    bigram_probs = get_bigram_probs(sentences, V, start_idx, end_idx, smoothing=0.1)

    # when we sample a fake sentence, we want to ensure not to
    # sample start token or end token
    sample_probs = np.ones(V)
    sample_probs[start_idx] = 0
    sample_probs[end_idx] = 0
    sample_probs /= sample_probs.sum()

    # test our model on real and fake sentences
    while True:
        # real sentence
        real_idx = np.random.choice(len(sentences))
        real = sentences[real_idx]

        # fake sentence
        fake = np.random.choice(V, size=len(real), p=sample_probs)

        print('REAL: %s\nSCORE: %f\n' % (get_words(real, idx2word), get_score(real, start_idx, end_idx, bigram_probs)))
        print('FAKE: %s\nSCORE: %f\n' % (get_words(fake, idx2word), get_score(fake, start_idx, end_idx, bigram_probs)))

        # input your own sentence
        custom = input('Enter your own sentence:\n')
        custom = custom.lower().split()

        # check that all tokens exist in word2idx (otherwise, we cannot get score)
        bad_sentence = False
        for token in custom:
            if token not in word2idx:
                bad_sentence = True
                break

        if bad_sentence:
            print('Sorry, you entered words that are not in the vocabulary')
        else:
            # converte sentence into list of indexes
            custom = [word2idx[token] for token in custom]
            print('SCORE: %f\n' % get_score(custom, start_idx, end_idx, bigram_probs))

        cont = input('Continue? [Y/n]')
        if cont in ('N', 'n'):
            break


if __name__ == '__main__':
    main()

