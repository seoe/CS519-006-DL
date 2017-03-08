# Loading the data
# Now, the data should contain these items:
# dict["train_data"]: a 10000 x 3072 matrix with each row being a training image (you can visualize the image by reshaping the row to 32x32x3
# dict["train_labels"]: a 10000 x 1 vector with each row being the label of one training image, label 0 is an airplane, label 1 is a ship.
# dict["test_data"]: a 2000 x 3072 matrix with each row being a testing image
# dict["test_labels]: a 2000 x 1 vector with each row being the label of one testing image, corresponding to test_data.

import cPickle
import numpy as np   # added by Kaibo
import matplotlib.pyplot as plt  # added by Kaibo
import time  # added by Kaibo

t0 = float(time.clock())
dict = cPickle.load(open("cifar_2class_py2.p","rb"))
t1 = float(time.clock())
print 'Loading time is %.4f s. \n' % (t1-t0)

for i in dict:
    print i, dict[i].shape


#-------add------
print type(dict), type(dict["train_data"]),',Shape:',dict["train_data"].shape, type(dict["train_data"][1]), ',Shape:',dict["train_data"][1].shape
#print dict
#print dict["train_data"]
#print dict["train_data"][1]
#print dict["train_data"][1][1]
true_label = ['airplane','ship']
X = dict["train_data"].reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
Y = np.array(dict["train_labels"])

for i in range(10):
	#img = np.reshape(dict["train_data"][i],(32,32,3), order='F')
	img = dict["train_data"][i].reshape((3,32,32)).transpose(1,2,0).astype("float")
	img2 = dict["train_data"][i].reshape((3,32,32)).transpose(1,2,0)
	img3 = dict["train_data"][i].reshape((32,32,3))

	label0 = true_label[dict["train_labels"][i][0]]
	print "train_data[%d]" % (i), dict["train_data"][i], label0
	#print img
	print 'img',img.shape,img.dtype
	print 'img2',img2.shape,img2.dtype
	#imgplot = plt.imshow(img)
	#plt.imshow(img)

	img0 = plt.figure()
	plt.suptitle('visualize CIFAR-10 with imgshow - img %d: %s' %(i,label0))

	plt.subplot(221)
	plt.title('reshape 32X32X3')
	plt.imshow(img3, interpolation='nearest')
	plt.axis('off')

	plt.subplot(222)
	plt.title('reshape.T+uint8,interp"none"')
	plt.imshow(img2, interpolation='none')
	#plt.imshow(X[i], interpolation='nearest')
	plt.axis('off')

	plt.subplot(223)
	plt.title('reshape.T+float,"none"')
	plt.imshow(img, interpolation='none')
	#plt.title('reshape.T+uint8,"spline36"')
	#plt.imshow(img2, interpolation='spline36')

	#plt.show()
	plt.subplot(224)
	plt.title('reshape.T+uint8,"sinc"')
	plt.imshow(img2, interpolation='sinc')

	plt.savefig('cmp_%d.png' %(i))
	plt.show()
	plt.close()

t2 = float(time.clock())
print 'Total time is %.4f s. \n' % (t2-t0)