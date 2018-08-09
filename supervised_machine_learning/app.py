import pickle
import numpy as np
import os
import json

import tornado.ioloop
import tornado.web

if not os.path.exists('mymodel.pkl'):
	exit('Cannot run without the model!')

with open('mymodel.pkl', 'rb') as f:
	model = pickle.load(f)

class MainHandler(tornado.web.RequestHandler):
	def get(self):
		self.write('Hello, Tornado!')

class PredictionHandler(tornado.web.RequestHandler):
	# predict one sample at a time
	def post(self):
		# print('body:', self.request.body)
		# print('arguments:', self.request.arguments)
		# will look like this:
		# body: three=four&one=two
		# arguments: {'three': ['four'], 'one': ['two']}

		params = self.request.arguments
		# x = np.array(map(float, params['input']))
		x = np.array([[float(e) for e in params['input']]]) # x: (1xD)
		y = int(model.predict(x)[0])
		self.write(json.dumps({'prediction': y}))
		self.finish()

if __name__ == '__main__':
	application = tornado.web.Application([
		(r"/", MainHandler),
		(r"/predict", PredictionHandler)
	])
	application.listen(8888)
	tornado.ioloop.IOLoop.current().start()