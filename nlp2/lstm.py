# Long Short-Term Memory (LSTM)

import numpy as np
import theano
import theano.tensor as T

from util import init_weight


class LSTM(object):

    def __init__(self, Mi, Mo, activation=T.nnet.relu):
        self.Mi = Mi
        self.Mo = Mo
        self.f  = activation

        # numpy init
        Wxi = init_weight(Mi, Mo)
        Whi = init_weight(Mo, Mo)
        Wci = init_weight(Mo, Mo)
        bi  = np.zeros(Mo)
        Wxf = init_weight(Mi, Mo)
        Whf = init_weight(Mo, Mo)
        Wcf = init_weight(Mo, Mo)
        bf  = np.zeros(Mo)
        Wxc = init_weight(Mi, Mo)
        Whc = init_weight(Mo, Mo)
        bc  = np.zeros(Mo)
        Wxo = init_weight(Mi, Mo)
        Who = init_weight(Mo, Mo)
        Wco = init_weight(Mo, Mo)
        bo  = np.zeros(Mo)
        h0  = np.zeros(Mo)
        c0  = np.zeros(Mo)

        # theano variables
        self.Wxi = theano.shared(Wxi)
        self.Whi = theano.shared(Whi)
        self.Wci = theano.shared(Wci)
        self.bi  = theano.shared(bi)
        self.Wxf = theano.shared(Wxf)
        self.Whf = theano.shared(Whf)
        self.Wcf = theano.shared(Wcf)
        self.bf  = theano.shared(bf)
        self.Wxc = theano.shared(Wxc)
        self.Whc = theano.shared(Whc)
        self.bc  = theano.shared(bc)
        self.Wxo = theano.shared(Wxo)
        self.Who = theano.shared(Who)
        self.Wco = theano.shared(Wco)
        self.bo  = theano.shared(bo)
        self.h0  = theano.shared(h0)
        self.c0  = theano.shared(c0)
        self.params = [
            self.Wxi, self.Whi, self.Wci, self.bi,
            self.Wxf, self.Whf, self.Wcf, self.bf,
            self.Wxc, self.Whc, self.bc,
            self.Wxo, self.Who, self.Wco, self.bo,
            self.h0,  self.c0,
        ]

    def recurrence(self, x_t, h_t1, c_t1):
        # x_t: (D, ), h_t: (M, ), c_t: (M, )
        # i_t: (M, ), f_t: (M, ), o_t: (M, )
        i_t = T.nnet.sigmoid(x_t.dot(self.Wxi) + h_t1.dot(self.Whi) + c_t1.dot(self.Wci) + self.bi) # input gate
        f_t = T.nnet.sigmoid(x_t.dot(self.Wxf) + h_t1.dot(self.Whf) + c_t1.dot(self.Wcf) + self.bf) # forget gate
        c_t = f_t * c_t1 + i_t * T.tanh(x_t.dot(self.Wxc) + h_t1.dot(self.Whc) + self.bc) # memory cell
        o_t = T.nnet.sigmoid(x_t.dot(self.Wxo) + h_t1.dot(self.Who) + c_t.dot(self.Wco) + self.bo) # output gate
        h_t = o_t * T.tanh(c_t)
        return h_t, c_t

    def output(self, x):
        # input x should be a matrix (T, D)
        # rows index time
        [h, c], _ = theano.scan(
            fn=self.recurrence,
            sequences=x,
            n_steps=x.shape[0],
            outputs_info=[self.h0, self.c0],
        )
        return h

