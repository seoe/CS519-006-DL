# Assignment #3: Keras CIFAR‐10 Image Classificaon
- [Assignment Description](#assignment-description)
- [Problems from keras version](#problems-from-keras-version)
    - [Dimension ordering problem](#dimension-ordering-problem)
    - [Model save/load problem](#model-saveload-problem)
    - [Layer pop problem](#layer-pop-problem)
    - [layer name in model.add](#layer-name-in-modeladd)
    - [Acc result display Problem](#acc-result-display-problem)



Model save/load problem
Layer pop problem
layer name in model.add

## Assignment Description
**Due** Mar 10 by 11:59pm **Points** 40 **Submitting** a file upload **File Types** zip

For this assignment, please either install Theano and Keras either on your own computer/server, or use the pelican server
(pelican.eecs.oregonstate.edu) where it has already been installed.

At your home directory, create a file named .theanorc and write these into it:
```zsh
[global]
floatX = float32
device = gpu0 # (or gpu1, depending on which gpu you would like to use)
```
With the `nvidia-smi`
command you can see the current GPU usage on the system. Try to pick an empty GPU.  
Then, put your CUDA into path. On pelican, this can be done by running in bash:
```zsh
export PATH=$PATH:/usr/local/eecsapps/cuda/cuda7.5.18/
bin
export LD_LIBRARY_PATH=/usr/local/eecsapps/cuda/cuda7.5.18/
lib64
```
If you use csh (default shell in pelican), please find your own way to set those environment variables. I have never used csh so I don't know how to. If you want to change your default shell to bash, you can do so at:
[https://secure.engr.oregonstate.edu:8000/teach.php?type=want_auth](https://secure.engr.oregonstate.edu:8000/teach.php?type=want_auth)  
by clicking "change unix shell" under "account tools".  
Now get the `cifar10_cnn.py` file from the Files tab in CANVAS, and run:
```zsh
python cifar10_cnn.py
```
(pelican only:) If each epoch is taking 2000+ seconds, then you didn't configure GPU correctly. If you configured correctly, each epoch should not
take more than 50 seconds (a 40 times speedup!).  
Now, go into cifar10_cnn.py, make the following tunings in the code:


1. Remove the dropout layer after the fully-connected layer (10 points). Save the model after training.

1. Load the model you saved at step 1 as initialization to the training. Add another fully connected layer with 512 filters at the second-to-last layer (before the classification layer) (10 points). Train and save the model (Hint: check the end of the assignment description to see how to pop out a layer).

1. Try to use an adaptive schedule to tune the learning rate, you can choose from RMSprop, Adagrad and Adam (Hint: you don't need to implement any of these, look at Keras documentation please) (10 points).

1. Try to tune your network in two other ways (10 points) (e.g. add/remove a layer, change the activation function, add/remove regularizer, change the number of hidden units) not described in the previous four. You can start from random initializations or previous results as you wish.

For each of the settings (1) - (4), please submit a PDF report your training loss, training error, validation loss and validation error. Draw 2 figures for each of the settings (1) - (4) (2 figures for each different tuning in (4)) with the x-axis being the epoch number, and y-axis being the loss/error, use 2 different lines in the same figure to represent training loss/validation loss, and training error/validation error.

**Name your file "firstname_lastname_hw3.pdf". Submit this pdf file on Canvas.**

### Tips for popping a layer:
In order to pop a layer, you can easily call model.layers.pop(). Layers is a Python list, which supports stack functions such as push and pop. However, a KNOWN problem is that if you don't save the popped layer, Python garbage collection will claim some memory which causes an error, so for the moment before we know what is the problem, let's call layer1 = model.layers.pop() instead of just calling model.layers.pop(), then it will work.

[***Back*** to TOP](#assignment-3-keras-cifar10-image-classificaon)

---

## Problems from keras version  

### Dimension ordering problem
From official documentation, there are:
> MaxPooling2D  Input shape  
> 4D tensor with shape: (samples, channels, rows, cols) if dim_ordering='th'  
> or   
> 4D tensor with shape: (samples, rows, cols, channels) if dim_ordering='tf'.

I believe keras has change the order of arguments in 4D shape from (samples, channels, rows, cols)/v-0.3.2 to (samples, rows, cols, channels)/v-1.2.2 if `dim_ordering='tf'`(in `~/.keras/keras.json/`). In older version, they are compatible with 'tf'.

So, for the keras version 1.2.2, I need to switch `dim_ordering` according to the backend.

|Layer|Group server|Pelican server|
|:-:|:-:|:-:|
|keras.\_\_version\_\_|1.2.2|0.3.2|
|MaxPooling2D Input shape('th' order)|(samples, **channels**, rows, cols)|(samples, channels, rows, cols)|
|MaxPooling2D Input shape('tf' order)|(samples, rows, cols, **channels**)|(samples, channels, rows, cols)|

So, the **`reason`** is, keras in TitanX is too recent (version `1.2.2`) for Assignment's code, which is only compatible to older version(like `0.3.2` in pelican), and the output dimension of AveragePooling2D function is different between the two versions.


[***Back*** to TOP](#assignment-3-keras-cifar10-image-classificaon)

---

### Model save/load problem

There are two ways to save trained model.  
One is to save/load **architecture** of model(**JSON**) and **weights**(**HDF5**) seperaterly.  
```python
model.to_json()
model.save_weights('m.h5')
...
model.load_weights('m.h5')
```
The other is to seve/load the whole model using `load_model`:
```python
from keras.models import load_model
...
model.save('m.h5')
...
load_model('m.h5')
```


[***Back*** to TOP](#assignment-3-keras-cifar10-image-classificaon)

---

### Layer pop problem

`l0 = model.layers.pop()` only pop the layer node, not the connection and output of model.   
So, we need to do like this:
```
model.layers.pop()
model.layers.pop()
model.outputs = [model.layers[-1].output]
model.layers[-1].outbound_nodes = []
```


[***Back*** to TOP](#assignment-3-keras-cifar10-image-classificaon)

---


### Layer name in model.add

check layer information with `model.summary()`  
In keras 1.2.2, if you load a model and add some layer, you need to give it a **new** name to avoid duplicate name because the loaded one may have been named as dense_1, and once you use .add(dense(512)), it will give 'dense_1' again. The right way should be `.add(dense(512,name="dense_new"))`


[***Back*** to TOP](#assignment-3-keras-cifar10-image-classificaon)

---

### Acc result display problem

keras 1.2.2 has another problem:
```
UserWarning: The "show_accuracy" argument is deprecated, instead you should pass the "accuracy" metric to the model at compile time:
`model.compile(optimizer, loss, metrics=["accuracy"])`
```
or the result will only has `loss` rather than `acc` after fitting. So I add the `metrics=["accuracy"]` as the last argument of model.compile.


[***Back*** to TOP](#assignment-3-keras-cifar10-image-classificaon)

---
