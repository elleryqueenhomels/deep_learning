# Discrete Hidden Markov Model (HMM) with scaling

import numpy as np
import matplotlib.pyplot as plt


def random_normalized(d1, d2):
    x = np.random.random((d1, d2))
    return x / x.sum(axis=1, keepdims=True)


class HMM(object):

    def __init__(self, M):
        self.M = M # number of hidden states

    def fit(self, X, max_iter=30, seed=123):
        if seed is not None:
            np.random.seed(seed)
        # train the HMM model using the Baum-Welch algorithm
        # a specific instance of the expectation-maximization algorithm

        # determine V, the vocabulary size
        # assume observations are already integers from 0..V-1
        # X is a jagged array of observed sequences
        V = max(max(x) for x in X) + 1
        N = len(X)
        M = self.M

        self.pi = np.ones(M) / M # initial state distribution
        self.A = random_normalized(M, M) # state transition matrix
        self.B = random_normalized(M, V) # output distribution

        print('initial A:', self.A)
        print('initial B:', self.B)

        costs = []
        for it in range(max_iter):
            alphas = []
            betas = []
            scales = []
            logP = np.zeros(N)

            for n in range(N):
                x = X[n]
                T = len(x)

                scale = np.zeros(T)
                alpha = np.zeros((T, M))
                alpha[0] = self.pi * self.B[:, x[0]]
                scale[0] = alpha[0].sum()
                alpha[0] /= scale[0]
                for t in range(1, T):
                    alpha_t_prime = alpha[t - 1].dot(self.A) * self.B[:, x[t]]
                    scale[t] = alpha_t_prime.sum()
                    alpha[t] = alpha_t_prime / scale[t]
                logP[n] = np.log(scale).sum()
                alphas.append(alpha)
                scales.append(scale)

                beta = np.zeros((T, M))
                beta[-1] = 1
                for t in range(T - 2, -1, -1):
                    beta[t] = self.A.dot(self.B[:, x[t + 1]] * beta[t + 1]) / scale[t + 1]
                betas.append(beta)

            cost = np.sum(logP)
            costs.append(cost)

            # now re-estimate pi, A, B
            self.pi = np.sum(alphas[n][0] * betas[n][0] for n in range(N)) / N

            den1 = np.zeros((M, 1))
            den2 = np.zeros((M, 1))
            a_num = np.zeros((M, M))
            b_num = np.zeros((M, V))
            for n in range(N):
                x = X[n]
                T = len(x)
                den1 += (alphas[n][:-1] * betas[n][:-1]).sum(axis=0, keepdims=True).T
                den2 += (alphas[n] * betas[n]).sum(axis=0, keepdims=True).T

                # numerator for A
                for i in range(M):
                    for j in range(M):
                        for t in range(T - 1):
                            a_num[i,j] += alphas[n][t,i] * betas[n][t+1,j] * self.A[i,j] * self.B[j, x[t+1]] / scales[n][t+1]

                # numerator for B
                # for j in range(M):
                #     for k in range(V):
                #         for t in range(T):
                #             if x[t] == k:
                #                 b_num[j,k] += alphas[n][t,j] * betas[n][t,j]
                for j in range(M):
                    for t in range(T):
                        b_num[j,x[t]] += alphas[n][t,j] * betas[n][t,j]

            self.A = a_num / den1
            self.B = b_num / den2

        print('A:', self.A)
        print('B:', self.B)
        print('pi:', self.pi)

        plt.plot(costs)
        plt.show()

    def log_likelihood(self, x):
        # return log P(x | model)
        # using the forward part of the forward-backward algorithm
        T = len(x)
        M = self.M
        scale = np.zeros(T)
        alpha = np.zeros((T, M))
        alpha[0] = self.pi * self.B[:, x[0]]
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0]
        for t in range(1, T):
            alpha_t_prime = alpha[t-1].dot(self.A) * self.B[:, x[t]]
            scale[t] = alpha_t_prime.sum()
            alpha[t] = alpha_t_prime / scale[t]
        return np.log(scale).sum()

    def log_likelihood_multi(self, X):
        return np.array([self.log_likelihood(x) for x in X])

    def get_state_sequence(self, x):
        # return the most likely state sequence given observed sequence x
        # using the Viterbi algorithm
        T = len(x)
        M = self.M
        delta = np.zeros((T, M))
        psi = np.zeros((T, M))
        delta[0] = np.log(self.pi) + np.log(self.B[:, x[0]])
        for t in range(1, T):
            for j in range(M):
                delta[t,j] = np.max(delta[t-1] + np.log(self.A[:,j]) + np.log(self.B[j,x[t]]))
                psi[t,j] = np.argmax(delta[t-1] + np.log(self.A[:,j]))

        # backtrack
        states = np.zeros(T, dtype=np.int32)
        states[T - 1] = np.argmax(delta[T - 1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

