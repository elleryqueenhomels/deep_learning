# Long Short-Term Memory (LSTM) with Batch Training

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

	def get_ht(self, xWxi_t, xWxf_t, xWxc_t, xWxo_t, h_t1, c_t1):
		i_t = T.nnet.sigmoid(xWxi_t + h_t1.dot(self.Whi) + c_t1.dot(self.Wci) + self.bi)
		f_t = T.nnet.sigmoid(xWxf_t + h_t1.dot(self.Whf) + c_t1.dot(self.Wcf) + self.bf)
		c_t = f_t * c_t1 + i_t * T.tanh(xWxc_t + h_t1.dot(self.Whc) + self.bc)
		o_t = T.nnet.sigmoid(xWxo_t + h_t1.dot(self.Who) + c_t.dot(self.Wco) + self.bo)
		h_t = o_t * T.tanh(c_t)
		return (h_t, c_t)

	def recurrence(self, xWxi_t, xWxf_t, xWxc_t, xWxo_t, is_start, h_t1, c_t1, h0, c0):
		output = T.switch(
			T.eq(is_start, 1),
			self.get_ht(xWxi_t, xWxf_t, xWxc_t, xWxo_t, h0, c0),
			self.get_ht(xWxi_t, xWxf_t, xWxc_t, xWxo_t, h_t1, c_t1)
		)
		h_t, c_t = output[0], output[1]
		return h_t, c_t

	def output(self, Xflat, startPoints):
		# Xflat should be (N*T, D)
		# calculate X after multiplying input weights
		XWxi = Xflat.dot(self.Wxi)
		XWxf = Xflat.dot(self.Wxf)
		XWxc = Xflat.dot(self.Wxc)
		XWxo = Xflat.dot(self.Wxo)

		[h, c], _ = theano.scan(
			fn=self.recurrence,
			sequences=[XWxi, XWxf, XWxc, XWxo, startPoints],
			n_steps=Xflat.shape[0],
			outputs_info=[self.h0, self.c0],
			non_sequences=[self.h0, self.c0]
		)

		return h

