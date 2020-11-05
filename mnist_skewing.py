#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:33:12 2019

@author: pec
"""

import numpy as np
from cumulative_spectral_gradient import CumulativeSpectralGradient
import matplotlib.pyplot as plt

def label_skewing(labels, num):
    result = np.copy(labels)
    indicies = [i for i in range(num)]
    for index in indicies:
        target_index = np.where(labels==index)
        target_index = list(np.ravel(np.array(target_index)))
        for skewed_index in list(target_index):
            another_label = np.random.choice(indicies, 1)[0]
            result[skewed_index] = another_label
    return result
            

if __name__ == '__main__':

    embedding_mode = 'tsne'
    
    np.random.seed(0)
    
    # prepare data
    from keras.datasets import mnist
    (x_train, y_train), (_, _) = mnist.load_data()
    num_of_pixel = 1
    for i in range(x_train.ndim - 1):
        num_of_pixel = num_of_pixel * np.shape(x_train)[i+1]
    x_train = x_train.reshape((np.shape(x_train)[0], num_of_pixel))
    y_train = np.ravel(y_train)
    
    # from sklearn import datasets
    # dataset = datasets.load_iris()
    # x_train = dataset.data
    # y_train = dataset.target
    
    label_number = np.size(np.unique(y_train))
    eigenvalues = np.zeros((label_number, label_number))
    csgs = np.zeros((label_number))
    montecarlo_sampling_size=100
    k_for_nearest_neighbor=3

    
    for num_of_skewed_labels in range(1, label_number+1):
        print('Now {} labels were skewed...'.format(num_of_skewed_labels))
        y_skewed = label_skewing(y_train, num_of_skewed_labels)
        
        model = CumulativeSpectralGradient()
        model.fit(x_train, y_skewed, montecarlo_sampling_size, k_for_nearest_neighbor, embedding_mode=embedding_mode)        
        model.show_result(only_graph=True, show=False, save=True, 
                        scatter_save_path='mnist_skewing/{}/{}_{}.png'.format(embedding_mode,embedding_mode,num_of_skewed_labels),
                        mds_save_path='mnist_skewing/{}/{}_mds{}.png'.format(embedding_mode,embedding_mode,num_of_skewed_labels))
        eigenvalues[num_of_skewed_labels-1, :] = model.eigenvalues
        csgs[num_of_skewed_labels-1] = model.csg
        

    horizontal_axis = [i for i in range(label_number)]
    for num_of_skewed_labels in range(label_number):
        plt.plot(horizontal_axis, eigenvalues[num_of_skewed_labels, :], marker='|', label=num_of_skewed_labels)
    plt.legend()
    plt.savefig('mnist_skewing/{}/{}_csgs.png'.format(embedding_mode,embedding_mode))
    plt.close()
    
    plt.plot(horizontal_axis, csgs, marker='D')
    plt.savefig('mnist_skewing/{}/{}_eigenvalues.png'.format(embedding_mode,embedding_mode))
    plt.close()