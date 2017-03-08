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

def loadnwriteValError():
    saveDir    = "./Figure/"
    loadDir    = "./Result/"
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    ques_list = [4.3,4.7,4.9,4.12,4.13]
    label_list = ['Q.4.1 Sig and Dropout for FC',
                'Q.4.2 Hidden units in FC',
                'Q.4.3 Local norm for activ',
                'Q.4.4 Add conv(128) layer',
                'Q.4.5 Data augmentation']
    plt.figure()
    color = 0
    for i in range(len(ques_list)):
        quesNo = ques_list[i]
        path = '%s_acc.csv' %(str(quesNo))
        mat = np.genfromtxt(loadDir+path, delimiter = ',')
        line, epoch = mat.shape
        plt.plot(range(1,epoch+1),1-mat[1], get_colour(color),label="%s" %(label_list[i]))
        plt.xlim(0, epoch)
        #plt.ylim(0.6, 0.9)
        plt.xlabel('epoch')
        plt.ylabel('Error')
        plt.legend(loc='upper right')
        # plt.legend(loc='lower right')
        plt.grid(True)
        color += 1
    plt.savefig(saveDir+"Fig_all_4.png")
    plt.clf()

def loadnwrite():
    saveDir    = "./Figure/"
    loadDir    = "./Result/"
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    quesNo = 4.14
    path1 = '%s_acc.csv' %(str(quesNo))
    path2 = '%s_loss.csv' %(str(quesNo))
    e_label_list = ['training erre','validation error','tmp']
    l_label_list = ['training loss','validation loss','tmp']

############ plot accuracy/error ##########
    mat = np.genfromtxt(loadDir+path1, delimiter = ',')
    line, epoch = mat.shape
    color = 0

    plt.figure()
    for i in xrange(line):
        plt.plot(range(1,epoch+1),1-mat[i], get_colour(color),label="%s" %(e_label_list[i]))
        plt.xlim(0, epoch)
        #plt.ylim(0.6, 0.9)
        plt.xlabel('epoch')
        plt.ylabel('Error')
        plt.legend(loc='upper right')
        # plt.legend(loc='lower right')
        plt.grid(True)
        color += 1
    plt.savefig(saveDir+"Fig_"+path1[:-4]+".png")
    plt.clf()

############ plot loss ##########
    mat = np.genfromtxt(loadDir+path2, delimiter = ',')
    line, epoch = mat.shape
    color = 0

    plt.figure()
    for i in xrange(line):
        plt.plot(range(1,epoch+1),mat[i], get_colour(color),label="%s" %(l_label_list[i]))
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
    # loadnwrite()
    loadnwriteValError()
