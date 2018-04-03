"""
by Eugene Seo (02/13/17)
CIFAR-10 Image Classification using Fully Connected Neural Network
Assignment #2, 2017 Winter CS519 Deep Learning

- Tuning Parameters
"""

from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import time
import matplotlib.pyplot as plt

import MLP_Skeleton as ms

def get_colour(color):
	'''
	b---blue   c---cyan  g---green    k----black
	m---magenta r---red  w---white    y----yellow
	'''
	color_set = ['r--','b--','m--','g--','c--','k--','y--']
	return color_set[color % 7]

def printError(train_loss, train_accuracy, test_loss, test_accuracy):
	print('\n    Train Loss:  {:.3f}    Train Acc.:  {:.2f}%'.format(
		train_loss,
		train_accuracy,
	))

	print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(
		test_loss,
		test_accuracy,
	))

def printError_Batch(epoch, b, total_loss):
	print(
        '\r[Epoch {}, mb {}]    Avg.Loss = {:.3f}'.format(
            epoch + 1,
            b + 1,
            total_loss,
        ),
        end='',
    )

def printPlot(nPlot, title, data, color, testId, testValue):
	plt.figure(nPlot)
	plt.title(title)
	plt.plot(data, get_colour(color),label="%s" % testId+str(testValue))
	plt.xlabel('epoch')
	if ("loss" in title):
		plt.ylabel("loss")
		plt.legend(loc='upper right')
	else:
		plt.ylabel("accuracy (%)")
		plt.legend(loc='lower right')
	plt.grid(True)
	plt.savefig(testId+title)


def TuneLearningRata(train_x, train_y, test_x, test_y,
	num_epochs, num_batches, hidden_units, learning_rate, momentum, l2_penalty, ep, filename):
	time.clock()
	t0 = float(time.clock())

	num_examples, input_dims = train_x.shape
	color = 0

	for lr in learning_rate:
		print("lr", lr)
		t2 = float(time.clock())

		trainErr = []
		testErr = []
		trainAcc = []
		testAcc = []

		mlp = ms.MLP(input_dims, hidden_units)

		for epoch in xrange(num_epochs):
			randList = np.arange(num_examples)
			np.random.shuffle(randList)
			batches = randList.reshape((num_batches, int(num_examples/num_batches)))

			train_loss = 0.0
			train_accuracy = 0.0

			for b in xrange(num_batches):
				total_loss = 0.0
				x_batch = train_x[batches[b],:]
				y_batch = train_y[batches[b],:]

				total_loss = mlp.train(x_batch, y_batch, lr, momentum, l2_penalty, ep) # tuning learning_rate
				if (total_loss == 0):
					break
				printError_Batch(epoch, b, total_loss)
				train_loss += total_loss
				train_accuracy += mlp.evaluate(x_batch, y_batch)[1]
				sys.stdout.flush()

			train_loss = train_loss / num_batches
			train_accuracy = train_accuracy / num_batches
			train_accuracy = 100. * train_accuracy
			trainErr.append(train_loss)
			trainAcc.append(train_accuracy)

			test_performance = mlp.evaluate(test_x, test_y)
			test_loss = test_performance[0]
			test_accuracy = test_performance[1]
			test_accuracy = 100. * test_accuracy
			testErr.append(test_loss)
			testAcc.append(test_accuracy)
			printError(train_loss, train_accuracy, test_loss, test_accuracy)

		printPlot(1, "train loss", trainErr, color, "lr-", lr)
		printPlot(2, "test loss", testErr, color, "lr-", lr)
		printPlot(3, "train accuracy", trainAcc, color, "lr-", lr)
		printPlot(4, "test accuracy", testAcc, color, "lr-", lr)
		color += 1
		print('\nbest accuracy: %.4f, ' % np.max(testAcc))
		t3 = float(time.clock())
		print('running time %.4f s, \n' % (t3-t2))

	plt.close()
	t1 = float(time.clock())
	print('running time_learning_rate %.4f s, \n' % (t1-t0))


def TuneMiniBatchSize(train_x, train_y, test_x, test_y,
	num_epochs, num_batches, hidden_units, learning_rate, momentum, l2_penalty, ep, filename):
	time.clock()
	t0 = float(time.clock())

	num_examples, input_dims = train_x.shape
	color = 0
	for mb in num_batches:
		print("mb", mb)
		t2 = float(time.clock())

		trainErr = []
		testErr = []
		trainAcc = []
		testAcc = []

		mlp = ms.MLP(input_dims, hidden_units)

		for epoch in xrange(num_epochs):
			randList = np.arange(num_examples)
			np.random.shuffle(randList)
			batches = randList.reshape((mb, int(num_examples/mb)))

			train_loss = 0.0
			train_accuracy = 0.0

			for b in xrange(mb):
				total_loss = 0.0
				x_batch = train_x[batches[b],:]
				y_batch = train_y[batches[b],:]

				total_loss = mlp.train(x_batch, y_batch, learning_rate, momentum, l2_penalty, ep)
				if (total_loss == 0):
					break
				printError_Batch(epoch, b, total_loss)
				train_loss += total_loss
				train_accuracy += mlp.evaluate(x_batch, y_batch)[1]
				sys.stdout.flush()

			train_loss = train_loss / mb
			train_accuracy = train_accuracy / mb
			train_accuracy = 100. * train_accuracy
			trainErr.append(train_loss)
			trainAcc.append(train_accuracy)

			test_performance = mlp.evaluate(test_x, test_y)
			test_loss = test_performance[0]
			test_accuracy = test_performance[1]
			test_accuracy = 100. * test_accuracy
			testErr.append(test_loss)
			testAcc.append(test_accuracy)
			printError(train_loss, train_accuracy, test_loss, test_accuracy)

		printPlot(5, "train loss", trainErr, color, "batch-", mb)
		printPlot(6, "test loss", testErr, color, "batch-", mb)
		printPlot(7, "train accuracy", trainAcc, color, "batch-", mb)
		printPlot(8, "test accuracy", testAcc, color, "batch-", mb)
		color += 1
		print('\nbest accuracy: %.4f, ' % np.max(testAcc))
		t3 = float(time.clock())
		print('running time %.4f s, \n' % (t3-t2))

	plt.close()
	t1 = float(time.clock())
	print('running time_num_batches %.4f s, \n' % (t1-t0))


def TuneHiddenUnits(train_x, train_y, test_x, test_y,
	num_epochs, num_batches, hidden_units, learning_rate, momentum, l2_penalty, ep, filename):
	time.clock()
	t0 = float(time.clock())

	num_examples, input_dims = train_x.shape
	color = 0
	for h in hidden_units:
		print("h", h)
		t2 = float(time.clock())

		trainErr = []
		testErr = []
		trainAcc = []
		testAcc = []
		mlp = ms.MLP(input_dims, h)

		for epoch in xrange(num_epochs):
			randList = np.arange(num_examples)
			np.random.shuffle(randList)
			batches = randList.reshape((num_batches, int(num_examples/num_batches)))

			train_loss = 0.0
			train_accuracy = 0.0

			for b in xrange(num_batches):
				total_loss = 0.0
				x_batch = train_x[batches[b],:]
				y_batch = train_y[batches[b],:]

				total_loss = mlp.train(x_batch, y_batch, learning_rate, momentum, l2_penalty, ep)
				if (total_loss == 0):
					break
				printError_Batch(epoch, b, total_loss)
				train_loss += total_loss
				train_accuracy += mlp.evaluate(x_batch, y_batch)[1]
				sys.stdout.flush()

			train_loss = train_loss / num_batches
			train_accuracy = train_accuracy / num_batches
			train_accuracy = 100. * train_accuracy
			trainErr.append(train_loss)
			trainAcc.append(train_accuracy)

			test_performance = mlp.evaluate(test_x, test_y)
			test_loss = test_performance[0]
			test_accuracy = test_performance[1]
			test_accuracy = 100. * test_accuracy
			testErr.append(test_loss)
			testAcc.append(test_accuracy)
			printError(train_loss, train_accuracy, test_loss, test_accuracy)

		printPlot(9, "train loss", trainErr, color, "hidden units-", h)
		printPlot(10, "test loss", testErr, color, "hidden units-", h)
		printPlot(11, "train accuracy", trainAcc, color, "hidden units-", h)
		printPlot(12, "test accuracy", testAcc, color, "hidden units-", h)
		color += 1
		print('\nbest accuracy: %.4f, ' % np.max(testAcc))
		t3 = float(time.clock())
		print('running time %.4f s, \n' % (t3-t2))

	plt.close()
	t1 = float(time.clock())
	print('running time_hidden_units %.4f s, \n' % (t1-t0))


def TuneMomentum(train_x, train_y, test_x, test_y,
	num_epochs, num_batches, hidden_units, learning_rate, momentum, l2_penalty, ep, filename):
	time.clock()
	t0 = float(time.clock())

	num_examples, input_dims = train_x.shape
	color = 0
	for m in momentum:
		print("mm", m)
		t2 = float(time.clock())

		trainErr = []
		testErr = []
		trainAcc = []
		testAcc = []

		mlp = ms.MLP(input_dims, hidden_units)

		for epoch in xrange(num_epochs):
			randList = np.arange(num_examples)
			np.random.shuffle(randList)
			batches = randList.reshape((num_batches, int(num_examples/num_batches)))

			train_loss = 0.0
			train_accuracy = 0.0

			for b in xrange(num_batches):
				total_loss = 0.0
				x_batch = train_x[batches[b],:]
				y_batch = train_y[batches[b],:]

				total_loss = mlp.train(x_batch, y_batch, learning_rate, m, l2_penalty, ep)
				if (total_loss == 0):
					break
				printError_Batch(epoch, b, total_loss)
				train_loss += total_loss
				train_accuracy += mlp.evaluate(x_batch, y_batch)[1]
				sys.stdout.flush()

			train_loss = train_loss / num_batches
			train_accuracy = train_accuracy / num_batches
			train_accuracy = 100. * train_accuracy
			trainErr.append(train_loss)
			trainAcc.append(train_accuracy)

			test_performance = mlp.evaluate(test_x, test_y)
			test_loss = test_performance[0]
			test_accuracy = test_performance[1]
			test_accuracy = 100. * test_accuracy
			testErr.append(test_loss)
			testAcc.append(test_accuracy)
			printError(train_loss, train_accuracy, test_loss, test_accuracy)

		printPlot(13, "train loss", trainErr, color, "momentum-", m)
		printPlot(14, "test loss", testErr, color, "momentum-", m)
		printPlot(15, "train accuracy", trainAcc, color, "momentum-", m)
		printPlot(16, "test accuracy", testAcc, color, "momentum-", m)
		color += 1
		print('\nbest accuracy: %.4f, ' % np.max(testAcc))
		t3 = float(time.clock())
		print('running time %.4f s, \n' % (t3-t2))

	plt.close()
	t1 = float(time.clock())
	print('running time_momentum %.4f s, \n' % (t1-t0))


def TuneL2Penalty(train_x, train_y, test_x, test_y,
	num_epochs, num_batches, hidden_units, learning_rate, momentum, l2_penalty, ep, filename):
	time.clock()
	t0 = float(time.clock())

	num_examples, input_dims = train_x.shape
	color = 0
	for lp in l2_penalty: # tuning num_batches
		print("l2", lp)
		t2 = float(time.clock())

		trainErr = []
		testErr = []
		trainAcc = []
		testAcc = []

		mlp = ms.MLP(input_dims, hidden_units)

		for epoch in xrange(num_epochs):

			randList = np.arange(num_examples)
			np.random.shuffle(randList)
			batches = randList.reshape((num_batches, int(num_examples/num_batches)))

			train_loss = 0.0
			train_accuracy = 0.0

			for b in xrange(num_batches):
				total_loss = 0.0
				x_batch = train_x[batches[b],:]
				y_batch = train_y[batches[b],:]

				total_loss = mlp.train(x_batch, y_batch, learning_rate, momentum, lp, ep)
				if (total_loss == 0):
					break
				printError_Batch(epoch, b, total_loss)
				train_loss += total_loss
				train_accuracy += mlp.evaluate(x_batch, y_batch)[1]
				sys.stdout.flush()

			train_loss = train_loss / num_batches
			train_accuracy = train_accuracy / num_batches
			train_accuracy = 100. * train_accuracy
			trainErr.append(train_loss)
			trainAcc.append(train_accuracy)

			test_performance = mlp.evaluate(test_x, test_y)
			test_loss = test_performance[0]
			test_accuracy = test_performance[1]
			test_accuracy = 100. * test_accuracy
			testErr.append(test_loss)
			testAcc.append(test_accuracy)
			printError(train_loss, train_accuracy, test_loss, test_accuracy)

		printPlot(17, "train loss", trainErr, color, "l2 penalty-", lp)
		printPlot(18, "test loss", testErr, color, "l2 penalty-", lp)
		printPlot(19, "train accuracy", trainAcc, color, "l2 penalty-", lp)
		printPlot(20, "test accuracy", testAcc, color, "l2 penalty-", lp)
		color += 1
		print('\nbest accuracy: %.4f, ' % np.max(testAcc))
		t3 = float(time.clock())
		print('running time %.4f s, \n' % (t3-t2))

	plt.close()
	t1 = float(time.clock())
	print('running time_l2_penalty %.4f s, \n' % (t1-t0))


def Validation(train_x, train_y, test_x, test_y,
	num_epochs, num_batches, hidden_units, learning_rate, momentum, l2_penalty, ep, filename):
	time.clock()
	t0 = float(time.clock())

	num_examples, input_dims = train_x.shape

	mlp = ms.MLP(input_dims, hidden_units)

	trainErr = []
	testErr = []
	trainAcc = []
	testAcc = []
	for epoch in xrange(num_epochs):
		print(epoch)

		randList = np.arange(num_examples)
		np.random.shuffle(randList)
		batches = randList.reshape((num_batches, int(num_examples/num_batches))) # tuning num_batches

		train_loss = 0.0
		train_accuracy = 0.0

		for b in xrange(num_batches):
			total_loss = 0.0
			x_batch = train_x[batches[b],:]
			y_batch = train_y[batches[b],:]

			total_loss = mlp.train(x_batch, y_batch, learning_rate, momentum, l2_penalty, ep)
			if (total_loss == 0):
				break
			printError_Batch(epoch, b, total_loss)
			train_loss += total_loss
			train_accuracy += mlp.evaluate(x_batch, y_batch)[1]
			sys.stdout.flush()

		train_loss = train_loss / num_batches
		train_accuracy = train_accuracy / num_batches
		train_accuracy = 100. * train_accuracy
		trainErr.append(train_loss)
		trainAcc.append(train_accuracy)

		test_performance = mlp.evaluate(test_x, test_y)
		test_loss = test_performance[0]
		test_accuracy = test_performance[1]
		test_accuracy = 100. * test_accuracy
		testErr.append(test_loss)
		testAcc.append(test_accuracy)
		printError(train_loss, train_accuracy, test_loss, test_accuracy)

	color = 0
	plt.figure(21)
	plt.title("loss")
	plt.plot(trainErr, get_colour(color),label="train")
	color += 1
	plt.plot(testErr, get_colour(color),label="test")
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.legend(loc='upper right')
	plt.grid(True)
	plt.savefig("%s_E" % filename)

	color = 0
	plt.figure(22)
	plt.title("accuracy")
	plt.plot(trainAcc, get_colour(color),label="train")
	color += 1
	plt.plot(testAcc, get_colour(color),label="test")
	plt.xlabel('epoch')
	plt.ylabel('accuracy (%)')
	plt.legend(loc='lower right')
	plt.grid(True)
	plt.savefig("%s_A" % filename)

	plt.close()
	t1 = float(time.clock())
	print('\nrunning time %.4f s, ' % (t1-t0))
	print('best accuracy: %.4f, \n' % np.max(testAcc))