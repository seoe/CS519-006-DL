# -*- coding: utf-8 -*-
"""
Created on Sat 02/05/2016 22:05:52
@author: Kaibo Liu
"""


from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# np.savetxt(newdir+"TestLoss_tune_lr+hidden_%d+mb_%d.csv" %(hidden_units,num_batches), loss_test_mat, delimiter = ',')


def get_colour(color):
    '''
    b---blue   c---cyan  g---green    k----black
    m---magenta r---red  w---white    y----yellow
'''
    color_set = ['r--','b--','m--','g--','c--','k--','y--']
    return color_set[color % 7]


def loadnwrite():
    path = 'TestAcc_tune_batch+lr_0.001+hidden_100.csv'
    mat = np.genfromtxt(path, delimiter = ',')
    line, epoch = mat.shape

    color = 0
    label_list = [100,500,1000,2000]

    plt.figure()
    for i in xrange(line):
        plt.plot(range(1,epoch+1),mat[i], get_colour(color),label="%s %s" %('batches ',str(label_list[i])))
        plt.xlim(0, epoch)
        plt.ylim(0.6, 0.9)
        plt.xlabel('epoch')
        plt.ylabel('Test Accuracy')
        plt.legend(loc='upper left')
        plt.grid(True)
        color += 1
    plt.savefig("new"+path[:-4]+".png")
    plt.clf()

if __name__ == '__main__':
    loadnwrite()
