"""
by Eugene Seo (02/13/17)
CIFAR-10 Image Classification using Fully Connected Neural Network
Assignment #2, 2017 Winter CS519 Deep Learning

- Training and Evaluating Fully Connected Neural Network
"""

from __future__ import division
from __future__ import print_function

import sys

import cPickle
import numpy as np
import time

import Tuning_Parameters as tp

# This is a class for a LinearTransform layer which takes an input
# weight matrix W and computes W x as the forward step
class LinearTransform(object):
    def __init__(self, W):
        self.W = W
        self.velocity = 0

    def forward(self, x):
        return np.dot(x, self.W)

    def backward(self, grad_output):
        return np.dot(grad_output, self.W.T)

    def update(self, delta, learning_rate=1.0, momentum=0.0, l2_penalty=0.0):
        regulization = l2_penalty * self.W
        delta = delta + regulization
        self.velocity = momentum * self.velocity - learning_rate * delta
        self.W += self.velocity

# This is a class for a ReLU layer max(x,0)
class ReLU(object):
    def __init__(self):
        self.x = 0

    def forward(self, x):
        self.x = x
        return np.maximum(0,x)

    def backward(self, grad_output):
        gradient = (self.x > 0) * 1.0
        gradient[np.where(self.x==0)] = 0.5
        return gradient*grad_output

# This is a class for a sigmoid layer followed by a cross entropy layer, the reason
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):
    def __init__(self):
        self.y = 0
        self.prob = 0

    def forward(self, x, y, w1, w2, l2_penalty=0.0):
        self.y = y
        self.prob = self.sigmoid(x)
        E = self.crossEntropy(self.prob, y, w1, w2, l2_penalty)
        return E

    def backward(self, grad_output):
        return (self.prob - self.y)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def crossEntropy(self, x, y, w1, w2, l2_penalty=0.0):
        E = -np.sum(y * np.log(x) + (1.0 - y) * np.log(1-x))/y.shape[0]
        E += 0.5 * l2_penalty * (np.linalg.norm(w1) + np.linalg.norm(w2))
        return E

    def evaluate(self, x, y, w1, w2, l2_penalty=0.0):
        performance = []
        prob = self.sigmoid(x)
        E = self.crossEntropy(prob, y, w1, w2, l2_penalty)
        performance.append(E)
        y_hat = 1 * (prob >= 0.5)
        accuracy = 1 - (np.sum(y_hat ^ y) / y.shape[0])
        performance.append(accuracy)
        return performance

# This is a class for the Multilayer perceptron
class MLP(object):
    def __init__(self, input_dims, hidden_units):
        self.input_dims = input_dims
        self.hidden_units = hidden_units
        self.output_units = 1
        self.W1 = np.random.normal(0, 0.01, (self.input_dims,self.hidden_units))
        self.w2 = np.random.normal(0, 0.01, (self.hidden_units,self.output_units))
        self.LT1 = LinearTransform(self.W1)
        self.ReLUf = ReLU()
        self.LT2 = LinearTransform(self.w2)
        self.SCE = SigmoidCrossEntropy()

    def train(
        self,
        x_batch,
        y_batch,
        learning_rate,
        momentum,
        l2_penalty,
        ep
    ):

        # feedforward
        f1 = self.LT1.forward(x_batch)
        z1 = self.ReLUf.forward(f1)
        f2 = self.LT2.forward(z1)
        E = self.SCE.forward(f2, y_batch, self.LT1.W, self.LT2.W, l2_penalty)

        # backpropagation
        gradient3 = self.SCE.backward(0)
        gradient2 = self.LT2.backward(gradient3)
        gradient1 = self.ReLUf.backward(gradient2)

        # weight update
        delta_w2 = np.dot(z1.T, gradient3)
        delta_w1 = np.dot(x_batch.T, gradient1)
        self.LT2.update(delta_w2, learning_rate, momentum, l2_penalty)
        self.LT1.update(delta_w1, learning_rate, momentum, l2_penalty)

        # check if it converged
        if (np.sum(abs(delta_w2))+np.sum(abs(delta_w1))) < ep:
            print("Converged")
            E = 0.0

        return E

    def evaluate(self, x, y, l2_penalty=0.0):
        f1 = self.LT1.forward(x)
        z1 = self.ReLUf.forward(f1)
        f2 = self.LT2.forward(z1)
        performance = self.SCE.evaluate(f2, y, self.LT1.W, self.LT2.W, l2_penalty)
        return performance


def normalizationBias(x):
    new_x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    bias = np.ones((x.shape[0],1)) # add bias terms
    new_x = np.append(new_x, bias, axis=1)
    return new_x


if __name__ == '__main__':
    time.clock()
    t0 = float(time.clock())

    data = cPickle.load(open('cifar_2class_py2.p', 'rb'))

    train_x = data['train_data']
    train_y = data['train_labels']
    test_x = data['test_data']
    test_y = data['test_labels']

    train_x = normalizationBias(train_x)
    test_x = normalizationBias(test_x)

    num_examples, input_dims = train_x.shape
    ep = 0.001
    num_epochs = 100

    # defalut parameter values
    num_batches = 1000
    hidden_units = 10
    momentum = 0.8
    l2_penalty = 0.001

    # tune learning_rate
    print("1. Tune learning_rate =========================")
    learning_rate  = [1e-06, 5e-06, 1e-05, 5e-05, 1e-04] # 1e-05
    tp.TuneLearningRata(train_x, train_y, test_x, test_y,
        num_epochs, num_batches, hidden_units, learning_rate, momentum, l2_penalty, ep,
        'lr1')
    learning_rate  = 1e-05

    # tune num_batches
    print("2. Tune num_batches ===========================")
    num_batches = [10, 50, 100, 500, 1000] # 50
    tp.TuneMiniBatchSize(train_x, train_y, test_x, test_y,
        num_epochs, num_batches, hidden_units, learning_rate, momentum, l2_penalty, ep,
        'mb1')
    num_batches = 50

    # tune hidden_units
    print("3. Tune hidden_units =========================")
    hidden_units = [10, 50, 100, 500, 1000] # 50
    tp.TuneHiddenUnits(train_x, train_y, test_x, test_y,
        num_epochs, num_batches, hidden_units, learning_rate, momentum, l2_penalty, ep,
        'hu1')
    hidden_units = 50

    # tune momentum
    print("4. Tune momentum =============================")
    momentum = [0.0, 0.6, 0.7, 0.8, 0.9] # 0.7
    tp.TuneMomentum(train_x, train_y, test_x, test_y,
        num_epochs, num_batches, hidden_units, learning_rate, momentum, l2_penalty, ep,
        'mm1')
    momentum = 0.7

    # tune l2_penalty
    print("5. Tune l2_penalty ===========================")
    l2_penalty = [0.0, 0.001, 0.01, 1, 10]
    tp.TuneL2Penalty(train_x, train_y, test_x, test_y,
        num_epochs, num_batches, hidden_units, learning_rate, momentum, l2_penalty, ep,
        'lp1')
    l2_penalty = 1

    t1 = float(time.clock())
    print('total tuning time %.4f s, \n' % (t1-t0))


    # Final model
    print("Final training and evaluation ================")
    num_epochs = 100
    learning_rate  = 1e-05
    num_batches = 50
    hidden_units = 50
    momentum = 0.7
    l2_penalty = 1

    tp.Validation(train_x, train_y, test_x, test_y,
        num_epochs, num_batches, hidden_units, learning_rate, momentum, l2_penalty, ep,
        'final')