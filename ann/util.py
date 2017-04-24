import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def get_clouds():
    Nclass = 500
    D = 2

    X1 = np.random.randn(Nclass, D) + np.array([0, -2])
    X2 = np.random.randn(Nclass, D) + np.array([2, 2])
    X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
    X = np.vstack([X1, X2, X3])
    Y = np.array([0]*Nclass + [1]*Nclass +[2]*Nclass)
    return X, Y

def get_spiral():
    # Idea: radius -> low...high
    #               (do not start at 0, otherwise points will be "mused" at origin)
    #       angle = low...high proportional to radius
    #               [0, 2pi/6, 4pi/6, ... , 10pi/6] --> [pi/2, pi/3 + pi/2, ... , 5pi/3 + pi/2]
    # x = r*cos(theta), y = r*sin(theta)

    radius = np.linspace(1, 10, 100)
    thetas = np.empty((6, 100))
    for i in range(6):
        start_angle = np.pi * i / 3.0 # 2pi/6 == pi/3
        end_angle = start_angle + np.pi / 2.0
        angles = np.linspace(start_angle, end_angle, 100)
        thetas[i] = angles

    # convert into cartesian coordinates
    X1 = np.empty((6, 100))
    X2 = np.empty((6, 100))
    for i in range(6):
        X1[i] = radius * np.cos(thetas[i])
        X2[i] = radius * np.sin(thetas[i])

    # inputs
    X = np.empty((600, 2))
    X[:,0] = X1.flatten()
    X[:,1] = X2.flatten()

    # add noise
    X += np.random.randn(*X.shape) * 0.5

    # targets
    Y = np.array([0]*100 + [1]*100 + [0]*100 + [1]*100 + [0]*100 + [1]*100)

    return X, Y

def get_data(limit=None):
    print("Reading in and transforming data...")
    df = pd.read_csv('../../python_test/data_set/MNIST_train.csv')
    data = df.as_matrix()
    np.random.shuffle(data)
    X = data[:, 1:] / 255.0 # data is from 0..255
    Y = data[:, 0]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y

def get_xor():
    X = np.zeros((200, 2))
    X[:50] = np.random.random((50, 2)) / 2 + 0.5 # (0.5-1, 0.5-1)
    X[50:100] = np.random.random((50, 2)) / 2 # (0-0.5, 0-0.5)
    X[100:150] = np.random.random((50, 2)) / 2 + np.array([[0, 0.5]]) # (0-0.5, 0.5-1)
    X[150:] = np.random.random((50, 2)) / 2 + np.array([[0.5, 0]]) # (0.5-1, 0-0.5)
    Y = np.array([0]*100 + [1]*100)
    return X, Y

def get_donut():
    N = 1000
    R_inner = 5
    R_outer = 10
    half_N = int(N / 2)

    # distance from origin is radius + random normal
    # angle theta is uniformly distributed between (0, 2pi)
    R1 = np.random.randn(half_N) + R_inner
    theta = 2*np.pi*np.random.random(half_N)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

    R2 = np.random.randn(half_N) + R_outer
    theta = 2*np.pi*np.random.random(half_N)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

    X = np.concatenate([ X_inner, X_outer ])
    Y = np.array([0]*half_N + [1]*half_N)
    return X, Y

def get_facial_data(balance_ones=False):
    # images are 48x48 = 2304 size vectors
    # N = 35887
    Y = []
    X = []
    first = True
    for line in open('../../python_test/data_set/fer2013.csv'):
        if first:
            first = False # skip the first line which is a header.
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(e) for e in row[1].split()])

    X, Y = np.array(X) / 255.0, np.array(Y)

    if balance_ones:
        # balance the 1 class
        X0, Y0 = X[Y!=1, :], Y[Y!=1]
        X1 = X[Y==1, :]
        X1 = np.repeat(X1, 9, axis=0)
        X = np.vstack([X0, X1])
        Y = np.concatenate((Y0, [1]*len(X1)))

    X, Y = shuffle(X, Y)

    return X, Y

