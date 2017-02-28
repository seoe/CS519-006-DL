# -*- coding: utf-8 -*-
"""
Created on Sat 02/27/2017 18:05:52
@author: Kaibo Liu
"""


from __future__ import division
from __future__ import print_function

import os
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
    saveDir    = "./Figure/"
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    path1 = '0_acc.csv'
    path2 = '0_loss.csv'
    label_list = ['train','validation','tmp']

############ plot accuracy ##########
    mat = np.genfromtxt(path1, delimiter = ',')
    line, epoch = mat.shape
    color = 0

    plt.figure()
    for i in xrange(line):
        plt.plot(range(1,epoch+1),mat[i], get_colour(color),label="%s" %(label_list[i]))
        plt.xlim(0, epoch)
        #plt.ylim(0.6, 0.9)
        plt.xlabel('epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.grid(True)
        color += 1
    plt.savefig(saveDir+"Fig_"+path1[:-4]+".png")
    plt.clf()

############ plot loss ##########
    mat = np.genfromtxt(path2, delimiter = ',')
    line, epoch = mat.shape
    color = 0

    plt.figure()
    for i in xrange(line):
        plt.plot(range(1,epoch+1),mat[i], get_colour(color),label="%s" %(label_list[i]))
        plt.xlim(0, epoch)
        #plt.ylim(0.6, 0.9)
        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.grid(True)
        color += 1
    plt.savefig(saveDir+"Fig_"+path2[:-4]+".png")
    plt.clf()

if __name__ == '__main__':
    loadnwrite()
