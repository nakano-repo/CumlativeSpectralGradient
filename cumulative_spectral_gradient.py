#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 09:49:39 2019

@author: Tomofumi Nakano
"""

import sys

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.spatial import distance
from sklearn.manifold import TSNE, MDS
import itertools
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import umap
from scipy.sparse.csgraph import connected_components #Need for using umap function


class CumulativeSpectralGradient():
    
    def _embedding(self, input_data):
        if(self.embedding_mode=='No_embedding'):
            print('No embedding...')
            self.embedded_data = input_data
        elif(self.embedding_mode=='tsne'):
            print('embedding with ' + self.embedding_mode + ' ...')
            self.embedded_data = TSNE(n_components=2, random_state=0).fit_transform(input_data)
        elif(self.embedding_mode=='umap'):
            print('embedding with ' + self.embedding_mode + ' ...')
            self.embedded_data = umap.UMAP().fit_transform(input_data)
        else:
            print('Error. \'embedding_mode\' must be No_embedding, tsne or umap.')
            sys.exit()


    def _calculate_dataframe(self, nbrs):
        distances, indices = nbrs.kneighbors(self.embedded_data)
        nearest_neighbor_is_same_class_or_not = np.zeros((len(self.labels), self.k_for_nearest_neighbor+1), dtype=int)
        
        # The first column of indices has theirselves indices
        for i in range(self.k_for_nearest_neighbor+1):
            nearest_neighbor_is_same_class_or_not[:,i] = (self.labels[indices][:,i]==self.labels[indices][:,0])
        sample_size_of_its_class = np.sum(nearest_neighbor_is_same_class_or_not, axis = 1) - 1
        
        # distances[:,-1] have radious of hypersphere
        # Eq. (4) of the original paper
        probability_list = sample_size_of_its_class / (self.k_for_nearest_neighbor * distances[:,-1] * distances[:,-1])
        df = pd.DataFrame({'data':list(self.embedded_data), 'labels':self.labels, 'probability':probability_list})
        
        # Normalize the sum of each class probability
        for i in range(self.label_number):
            normalization_constant = df[df.labels==i].probability.sum()
            df.loc[df.labels==i, 'probability'] = df[df.labels==i]['probability'] / normalization_constant
        return df
    
    
    def _calculate_similarity_matrix(self):
        nbrs = NearestNeighbors(n_neighbors=self.k_for_nearest_neighbor+1).fit(self.embedded_data) 
        df = self._calculate_dataframe(nbrs)
        
        self.similarity_matrix = np.zeros((self.label_number, self.label_number))
        
        # Montecarlo sampling
        # Eq. (3) of the original paper
        for i in range(self.label_number):
            for j in range(self.label_number):
                df_of_class_i = df[df.labels==i]
                sample_indices_categorical = np.random.multinomial(1, df_of_class_i.probability, self.montecarlo_sampling_size)
                sample_indices = np.where(sample_indices_categorical == 1)[1]
                
                probability_given_j = 0
                
                # Note sample_indices has different indices of the original dataframe (df), thus iloc was used
                # nbrs model outputs indices of the original dataframe (df)
                for index in list(sample_indices):
                    distances, indices = nbrs.kneighbors([list(df_of_class_i.iloc[index]['data'])])
                    sample_size_of_class_j = np.sum(np.array(df.loc[indices[0]].labels)==j)
                    probability_given_j = probability_given_j + sample_size_of_class_j / (self.k_for_nearest_neighbor * distances[0,-1] * distances[0,-1])
                    
                self.similarity_matrix[i, j] = probability_given_j / self.montecarlo_sampling_size

    
    def _calculate_weighted_adjacency_matrix(self):
        self.weighted_adjacency_matrix = np.zeros((self.label_number, self.label_number))
        # calculate only upperside of weighted_adjacency_matrix
        combinations = list(itertools.combinations_with_replacement(range(self.label_number),2))
        # Eq. (5) of the original paper
        for element in combinations:
            self.weighted_adjacency_matrix[element] = 1 - distance.braycurtis(self.similarity_matrix[element[0]], 
                                                                              self.similarity_matrix[element[1]])
        # caluclate symmetric matrix
        triu = np.triu(self.weighted_adjacency_matrix)
        self.weighted_adjacency_matrix = triu + triu.T - np.diag((np.diag(triu)))

    
    def _calculate_laplacian_matrix(self):
        degree_matrix = np.diag(np.sum(self.weighted_adjacency_matrix, axis=0))
        self.laplacian_matrix = degree_matrix - self.weighted_adjacency_matrix

    
    def _calculate_eigenvalues(self):
        self.eigenvalues, self.eigenvector = eigh(self.laplacian_matrix)
        eigen_id = np.argsort(self.eigenvalues)
        self.eigenvalues = self.eigenvalues[eigen_id]
        self.eigenvector = self.eigenvector[:,eigen_id]

    
    def _calculate_csg(self):
        eigenvalue_gradients = np.zeros((self.label_number - 1))
        # Eq. (6) of the original paper
        for i in range(self.label_number - 1):
            eigenvalue_gradients[i] = (self.eigenvalues[i+1] - self.eigenvalues[i]) / (self.label_number - i)
        eigenvalue_gradients_df = pd.Series(eigenvalue_gradients)
        # Eq. (7) of the original paper
        self.csg = sum(eigenvalue_gradients_df.cummax())
        
    
    def fit(self, input_data, labels, montecarlo_sampling_size=100, k_for_nearest_neighbor=3, embedding_mode='TSNE'):
        self.labels = labels
        self.label_number = len(np.unique(labels))
        self.montecarlo_sampling_size = montecarlo_sampling_size
        self.k_for_nearest_neighbor = k_for_nearest_neighbor
        self.embedding_mode = embedding_mode
        
        self._embedding(input_data)
        self._calculate_similarity_matrix()
        self._calculate_weighted_adjacency_matrix()
        self._calculate_laplacian_matrix()
        self._calculate_eigenvalues()
        self._calculate_csg()
        
        return self
    
    
    def multi_dimensional_scaling(self):
        self.dissimilarity_matrix = 1 - self.weighted_adjacency_matrix
        mds = MDS(n_components=2, dissimilarity="precomputed")
        mds_coordinate = mds.fit_transform(self.dissimilarity_matrix)
        plt.axes().set_aspect('equal')

        for l in set(self.labels):
            plt.scatter(mds_coordinate[l, 0], mds_coordinate[l, 1], label=l)
    
        plt.xlim(-1.7, 1.7)
        plt.ylim(-1.7, 1.7)
        plt.legend()
        
        
    def show_result(self, only_graph=False, show=True, save=False, scatter_save_path=None, mds_save_path=None):
        plt.scatter(self.embedded_data[:,0], self.embedded_data[:,1], c=self.labels, cmap=cm.tab10)
        plt.colorbar()
        if save:
            plt.savefig(scatter_save_path)
        if show:
            plt.show()
        plt.close()
        
        if not only_graph:
            np.set_printoptions(precision=4, suppress=True)
            print('S=')
            print(self.similarity_matrix)
            print('\nW=')
            print(self.weighted_adjacency_matrix)
            print('\nL=')
            print(self.laplacian_matrix)
            print('\neigenvalues')
            print(self.eigenvalues)
            print('\nCSG c-measure: ' + str(self.csg))
        
        self.multi_dimensional_scaling()
        if save:
            plt.savefig(mds_save_path)
        if show:
            plt.show()
        plt.close()

    
if __name__ == '__main__':
    
    np.random.seed(0)
    
    from sklearn import datasets
    dataset = datasets.load_iris()
    #dataset = datasets.load_digits()
    
    montecarlo_sampling_size=100
    k_for_nearest_neighbor=3
    model = CumulativeSpectralGradient()
    
    import time
    t_before = time.time()
    model.fit(dataset.data, dataset.target, montecarlo_sampling_size, k_for_nearest_neighbor, embedding_mode='umap')
    t_after = time.time()
    elapsed_time = t_after - t_before
    
    model.show_result(only_graph=False)
    
    print('elapsed time: ' + str(elapsed_time))
    
    