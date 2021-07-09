#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 17:46:24 2021

@author: mdreveto
"""

import numpy as np




def updateTransitionMatrix( old_edges, new_edges, transitionsMatrix ):
    
    for edge in old_edges :
        if edge in new_edges:
            transitionsMatrix[ edge[0], edge[1], 1, 1 ] += 1
            transitionsMatrix[ edge[1], edge[0], 1, 1 ] += 1
        else:
            transitionsMatrix[ edge[0], edge[1], 1, 0 ] += 1
            transitionsMatrix[ edge[1], edge[0], 1, 0 ] += 1
    
    for edge in new_edges:
        if edge not in old_edges :
            transitionsMatrix[ edge[0], edge[1], 0, 1 ] += 1
            transitionsMatrix[ edge[1], edge[0], 0, 1 ] += 1
    
    return transitionsMatrix



def parameterEstimation( transitionsMatrix, nodesInEachCluster, n , t, K = 2):
    """
    Return estimators for the parameters Pin and Pout (Markov transition probabilities) as well as piin and piout (initial distribution) 
    given a clustering (nodesInEachCluster) (this can be an estimated clustering)
    """
    Pin = np.zeros( (2,2) , dtype=np.float64 )
    Pout = np.zeros( (2,2) , dtype=np.float64 )
    count = 0
    n0 = 0
    n1 = 0
    for cluster in range(K):
        for i in nodesInEachCluster[cluster]:
            for j in nodesInEachCluster[cluster]:
                if (j!=i):
                    n1 += transitionsMatrix[ i, j, 1, 0 ] + transitionsMatrix[ i, j, 1, 1 ]
                    n0 += t-1 - (transitionsMatrix[ i, j, 0, 1] + transitionsMatrix[ i, j, 1, 0]  + transitionsMatrix[ i, j, 1, 1]) + transitionsMatrix[ i, j, 0, 1 ]

                    Pin[ 0,1 ] += transitionsMatrix[ i, j, 0, 1] 
                    Pin[ 1,0 ] += transitionsMatrix[ i, j, 1, 0] 
                    Pin[ 1,1 ] += transitionsMatrix[ i, j, 1, 1] 
                    
                    count += 1
    #print(Pin)
    if( n0!=0 ):
        Pin[0,1] /= n0
        Pin[0,0] = 1 - Pin[0,1]
    if (n1 != 0 ):
        Pin[1,0] /= n1
        Pin[1,1] /= n1
    
    count = 0
    n0 = 0
    n1 = 0
    for cluster in range(K):
        for i in nodesInEachCluster[cluster]:
            otherClustersNodes = [dummy for dummy in range(n) if dummy not in nodesInEachCluster[cluster] ]
            for j in otherClustersNodes:
                n1 +=  transitionsMatrix[ i, j, 1, 0 ] + transitionsMatrix[ i, j, 1, 1 ]
                n0 += t-1 - (transitionsMatrix[ i, j, 0, 1] + transitionsMatrix[ i, j, 1, 0]  + transitionsMatrix[ i, j, 1, 1]) + transitionsMatrix[ i, j, 0, 1 ]
                
                Pout[ 0,1 ] += transitionsMatrix[ i, j, 0, 1] 

                Pout[ 1,0 ] += transitionsMatrix[ i, j, 1, 0] 
                Pout[ 1,1 ] += transitionsMatrix[ i, j, 1, 1] 
                
                count += 1
    #print(Pout)
    if( n0!=0 ):
        Pout[0,1] /= n0
        Pout[0,0] = 1 - Pout[0,1]
    if (n1 != 0 ):
        Pout[1,0] /= n1
        Pout[1,1] /= n1
    if count == 0:
        Pout[0,0] = 1
        Pout[1,0] = 1
        
    return ( Pin, Pout )




def log_likelihood_ratios( Pin, Pout ):
    
    l = np.zeros( (2,2) )
    if( Pin[0,0] * Pout[0,0] != 0 ):
        l[0,1] = np.log( Pin[0,0] / Pout[0,0] )

    if( Pin[0,1] * Pout[0,1] != 0 ):
        l[0,1] = np.log( Pin[0,1] / Pout[0,1] )
    if(Pin[1,0] * Pout[1,0] != 0):
        l[1,0] = np.log( Pin[1,0] / Pout[1,0] )
    if (Pin[1,1] * Pout[1,1] != 0):
        l[1,1] = np.log( Pin[1,1] / Pout[1,1] )
    
    if Pin[0,0] == 0:
        l[0,0] = -99
    if Pout[0,0] == 0:
        l[0,0] = 99
    if Pin[0,0] == 0 and Pout[0,0] == 0:
        l[0,0] = 0
    
    if Pin[0,1] == 0:
        l[0,1] = - 99
    if Pout[0,1] == 0:
        l[0,1] = + 99
    if (Pin [0,1] == 0 and Pout[0,1] == 0):
        l[0,1] = 0
    if Pin[1,0] == 0:
        l[1,0] = - 99
    if Pout[1,0] == 0:
        l[1,0] = + 99
    if (Pin [1,0] == 0 and Pout[1,0] == 0):
        l[1,0] = 0
    
    
    return l