'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).

Note: the data was pickled with Python 2, and some encoding issues might prevent you
from loading it in Python 3. You might have to load it in Python 2,
save it in a different format, load it in Python 3 and repickle it.
'''

from __future__ import print_function
from keras.datasets import cifar10
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import load_model
import numpy as np
import time
import os
import json

'''
# function for early version of keras
def loadmodel(path):
    # model reconstruction from JSON:
    from keras.models import model_from_json
    json_file = open(path+'.json', 'r')
    json_model = json_file.read()
    json_file.close()
    model = model_from_json(json_model)

    # load weights into new model
    model.load_weights(path+'.h5')  # loads a HDF5 file '*-1_cifar10_model.h5' for weights
    return model
'''

quesNo = 4.15  # to mark the question number, for data saving

saveDir    = "./Result/"
if not os.path.exists(saveDir):
    os.makedirs(saveDir)

t00 = t0 = float(time.clock())

# load_path = saveDir+'%s_cifar10_model' %(str(quesNo-1))
# L_model = loadmodel(saveDir+'%s_cifar10_model' %(str(quesNo-1)))
# model.summary()
# t1 = float(time.clock())
# print("Loaded model_%s from disk, model loading time is %.4f s." %(str(quesNo-1), t1-t0))

batch_size = 32
nb_classes = 10
nb_epoch = 80
data_augmentation = True  # true for 4.5

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

t1 = float(time.clock())
print ('Loading time is %.4f s.' % (t1-t0))
t0 = t1
model = Sequential()
model.add(Convolution2D(48, 3, 3, border_mode='same', input_shape=(3, 32, 32)))
model.add(Activation('relu'))
model.add(Convolution2D(48, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Convolution2D(96, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(96, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Convolution2D(192, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(192, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))

print ('Modeling time is %.4f s.' % (t1-t0))
t0 = t1

# let's train the model using SGD + momentum (how original).
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])
# adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])


print ('outNo:',quesNo)
t1 = float(time.clock())
print ('Compiling time is %.4f s.' % (t1-t0))
t0 = t1

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    hist = model.fit(X_train, Y_train, batch_size=batch_size,
              nb_epoch=nb_epoch, show_accuracy=True,
              validation_data=(X_test, Y_test), shuffle=True)
else:
    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    # fit the model on the batches generated by datagen.flow()
    hist = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch, show_accuracy=True,
                        validation_data=(X_test, Y_test),
                        nb_worker=1)

t1 = float(time.clock())
print ('Training time is %.4f s.' % (t1-t0))
t0 = t1

print ('outNo:',quesNo)
model.save(saveDir+'%s_cifar10_model.h5' %(str(quesNo)))  # creates a HDF5 file 'my_model.h5'
'''
# serialize model to JSON
json_model = model.to_json()
with open(saveDir+'%s_cifar10_model.json' %(str(quesNo)), "w") as json_file:
#    json_file.write(json.dumps(json.loads(model_json), indent=4))
#    json.dump(json_model,json_file)
    json_file.write(json_model)
# serialize weights to HDF5
model.save_weights(saveDir+'%s_cifar10_model.h5' %(str(quesNo)),overwrite=True)  # creates a HDF5 file for weights '*_cifar10_model.h5'
'''
print("Saved %s_model to disk" %(str(quesNo)))
loss_mat, acc_mat = [],[]
tmp = hist.history
print (type(tmp))
print (tmp.keys())
loss_mat.append(tmp['loss'])
loss_mat.append(tmp['val_loss'])
acc_mat.append(tmp['acc'])
acc_mat.append(tmp['val_acc'])

np.savetxt(saveDir+"%s_loss.csv" %(str(quesNo)), loss_mat, delimiter = ',')
np.savetxt(saveDir+"%s_acc.csv" %(str(quesNo)), acc_mat, delimiter = ',')

t1 = float(time.clock())
print ('Saving time is %.4f s.' % (t1-t0))
print ('Total time is %.4f s.' % (t1-t00))
