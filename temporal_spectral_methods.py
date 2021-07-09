#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 11:41:41 2021

@author: mdreveto
"""

"""
Created on Wed Jun  9 16:17:58 2021

@author: mdreveto
"""


import numpy as np
import scipy as sp
from tqdm import tqdm
import networkx as nx
from sklearn.cluster import SpectralClustering, KMeans


import estimation_parameters as estimation_parameters




def staticSpectralClustering( adjacencyMatrix, n_clusters = 2 ):
    sc = SpectralClustering(n_clusters = n_clusters, affinity='precomputed', assign_labels='discretize')
    labels_pred_spec = sc.fit_predict( adjacencyMatrix ) + np.ones( adjacencyMatrix.shape[0] )
    return labels_pred_spec.astype('int8')




def makeGeneralizedAdjacencyMatrix( adjacencyMatrix, sigma = 1/2 ):
    """
    Return the generalized  normalized adjacency matrix D^(-sigma) * A * D^(sigma-1)
    """
    n = adjacencyMatrix.shape[0]
    D = np.sum( adjacencyMatrix, axis=0 )

    
    D1 = sp.sparse.lil_matrix( ( n, n ) ) #Will correspond to D^{-sigma}
    D1_vector = ( np.power( abs( D ), - float( sigma ) ) )
    for i in range(n):
        D1[i,i] = D1_vector[i]
    D1 = sp.sparse.dia_matrix( D1 )
    
    D2 = sp.sparse.lil_matrix( ( n, n ) ) #will correspond to D^{sigma-1}
    D2_vector = ( np.power( abs( D ), float( sigma - 1 ) ) ) 
    for i in range(n):
        D2[i,i] = D2_vector[i]
    D2 = sp.sparse.dia_matrix( D2 )

    return D1 @ sp.sparse.csr_matrix( adjacencyMatrix ) @ D2



def staticSpectralClustering_personal_implementation( adjacencyMatrix, n_clusters = 2):
    
    L = sp.sparse.eye( adjacencyMatrix.shape[0] ) - makeGeneralizedAdjacencyMatrix( adjacencyMatrix, sigma = 1/2 )
    
    vals, vecs = np.linalg.eigh( L.todense() )    
    vecs_with_constant_one = vecs[:,0:n_clusters ]
    kmeans = KMeans( n_clusters = n_clusters, random_state=0 ).fit( vecs_with_constant_one )
    labels_pred = kmeans.labels_ + np.ones( adjacencyMatrix.shape[ 0 ] )
    
    return labels_pred.astype(int)


def temporalWeightedSpectralClustering( N, temporal_edges, K = 2 ):
    T = len( temporal_edges )
    labelsPred = np.zeros( [ N, T ], dtype = int )


    A = np.zeros( ( N, N ) ) #aggregated (weigthed) adjacency matrix
    
    for t in tqdm( range( T ) ):
        for edge in temporal_edges[t]:
            A[ edge[0], edge[1] ] += 1
            A[ edge[1], edge[0] ] += 1
        
        labelsPred[ :, t ] = staticSpectralClustering( adjacencyMatrix = A, n_clusters = K )
        #M = sp.sparse.csc_matrix( A )
        #labelsPred[ :, t ] = staticSpectralClustering_adjacency_matrix( M, n_clusters = K )
    return labelsPred



def temporalSpectralClustering_knownParameters_first_formulation( N, temporal_edges, alpha , beta, K = 2, initialisation = 'random', labels_oracle = [] ):
    
    T = len( temporal_edges )
    labelsPred = np.zeros( [ N, T ], dtype = int )

    if (initialisation == 'random'):
        labelsPred[:,0] = np.random.randint( 1, K+1, size  = N ) #initialization at random    
    elif (initialisation == 'oracle'):
        labelsPred[:,0] = labels_oracle
    
    Apers = np.zeros( ( N, N ) )
    Anew = np.zeros( ( N, N ) )
    
    for t in tqdm( range(1, T) ):
        for edge in temporal_edges[t]:
            Anew[ edge[0], edge[1] ] += 1
            Anew[ edge[1], edge[0] ] += 1
            if edge in temporal_edges[t-1]:
                Apers[ edge[0], edge[1] ] += 1
                Apers[ edge[1], edge[0] ] += 1
        
        M = alpha * Anew + beta * Apers
        
        labelsPred[ :, t ] = staticSpectralClustering( adjacencyMatrix = M, n_clusters = K )

    return labelsPred




def temporalSpectralClustering_knownParameters( N, temporal_edges, alpha , beta, K = 2, initialisation = 'random', labels_oracle = [], useTqdm = True ):
    
    T = len( temporal_edges )
    labelsPred = np.zeros( [ N, T ], dtype = int )

    if (initialisation == 'random'):
        labelsPred[:,0] = np.random.randint( 1, K+1, size  = N ) #initialization at random    
    elif (initialisation == 'oracle'):
        labelsPred[:,0] = labels_oracle
    
    Apers = np.zeros( ( N, N ) )
    Anew = np.zeros( ( N, N ) )
    
    if (useTqdm):
        loop = tqdm( range( 1, T) )
    else:
        loop = range( 1, T )
    
    for t in loop:
        for edge in temporal_edges[t]:
            if edge not in temporal_edges[t-1]:
                Anew[ edge[0], edge[1] ] += 1
                Anew[ edge[1], edge[0] ] += 1
            else:
                Apers[ edge[0], edge[1] ] += 1
                Apers[ edge[1], edge[0] ] += 1
        
        M = alpha * Anew + beta * Apers

        labelsPred[ :, t ] = staticSpectralClustering( adjacencyMatrix = M, n_clusters = K )
        #M = sp.sparse.csc_matrix( M )
        #labelsPred[ :, t ] = staticSpectralClustering_adjacency_matrix( M, n_clusters = K )

    return labelsPred
    


