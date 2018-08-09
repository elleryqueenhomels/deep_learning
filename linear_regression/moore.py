import re
import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []

non_decimal = re.compile(r'[^\d]+')

for line in open('moore.csv'):
	r = line.split('\t')

	x = int(non_decimal.sub('', r[2].split('[')[0]))
	y = int(non_decimal.sub('', r[1].split('[')[0]))
	X.append(x)
	Y.append(y)

X = np.array(X)
Y = np.array(Y)

plt.scatter(X, Y)
plt.show()

Y = np.log(Y)
xSum = X.sum()
xMean = X.mean()
yMean = Y.mean()

a = (X.dot(Y) - yMean * xSum) / (X.dot(X) - xMean * xSum)
b = yMean - xMean * a

Yhat = a * X + b

plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

d1 = Y - Yhat
d2 = Y - yMean
r2 = 1 - d1.dot(d1) / d2.dot(d2)

#log(tc) = a * year + b
#tc = exp(a * year) * exp(b)
#2 * tc = 2 * exp(a * year) * exp(b)
#       = exp(ln2) * exp(a * year) * exp(b)
#       = exp(a * year + ln2) * exp(b)
#exp(a * year2) * exp(b) = exp(a * year1 + ln2) * exp(b)
#a * year2 = a * year1 + ln2
#year2 = year1 + ln2/a
#delta(year) = year2 - year1 = ln2/a

print('a =', a, ', b =', b)
print('r-squared =', r2)
print('Transistors count doubles every', np.log(2) / a, 'years')
