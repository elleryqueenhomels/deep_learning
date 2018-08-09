import requests
import numpy as np
import matplotlib.pyplot as plt
from util import get_data

# make a prediction from our own server!
# in reality this could be coming from any client.

X, Y = get_data()
N = len(Y)
while True:
	i = np.random.choice(N)
	r = requests.post("http://localhost:8888/predict", data={'input': X[i]})
	j = r.json()
	print('\nprediction:', j)
	print('target:', Y[i])

	plt.imshow(X[i].reshape(28, 28), cmap='gray')
	plt.show()

	response = input('Continue? (Y/N)\n')
	if response in ('n', 'N'):
		break
