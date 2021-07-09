#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 10:26:09 2021

@author: mdreveto
"""

import numpy as np
import random as random
from tqdm import tqdm

def makeMSBM_temporal_edges( N, T, initialDistributionRateMatrix, TransitionRateMatrix, nodesLabels, tqdm_ = False ):
    temporal_edges = dict( )
    for t in range( T ):
        temporal_edges[t] = [ ]
        
    if tqdm_:
        loop = tqdm( range( N ) )
    else:
        loop = range( N )
        
    for i in loop:
        for j in range( N ):
            temporal_edges = add_temporal_edges_of_a_given_nodepair( temporal_edges, i, j, initialDistributionRateMatrix[ nodesLabels[i] - 1 , nodesLabels[j] - 1 ] , TransitionRateMatrix[ nodesLabels[i] -1, nodesLabels[j] -1 ] )
            
    return temporal_edges


def make_degree_corrected_SBM_Markov_temporal_edges( N, T, initialDistributionRateMatrix, TransitionRateMatrix, nodesLabels, Theta, tqdm_ = False ):
    temporal_edges = dict( )
    for t in range( T ):
        temporal_edges[t] = [ ]
        
    if tqdm_:
        loop = tqdm( range( N ) )
    else:
        loop = range( N )
        
    for i in loop:
        for j in range( N ):
            initialDIstribution_ij = initialDistributionRateMatrix[ nodesLabels[i] - 1 , nodesLabels[j] - 1 ]
            initialDIstribution_ij = Theta[ i ] * Theta[ j ] * initialDIstribution_ij
            #initialDIstribution_ij[ 0 ] = 1 - initialDIstribution_ij
            
            transitionMatrix_ij = TransitionRateMatrix[ nodesLabels[i] -1, nodesLabels[j] -1 ].copy( )
            transitionMatrix_ij[ 0, 1 ] = Theta[i] * Theta[j] * transitionMatrix_ij[ 0, 1 ]
            transitionMatrix_ij[ 0, 0 ] = 1 - transitionMatrix_ij[ 0, 1 ]
            
            temporal_edges = add_temporal_edges_of_a_given_nodepair( temporal_edges, i, j, initialDIstribution_ij , transitionMatrix_ij )
            
    return temporal_edges



def makeTimeSerie( T, initialDistribution, TransitionMatrix ):
    x = np.zeros( T )
    x[0] = ( random.random() < initialDistribution )*1
    for i in range( 1, T ):
        if x[i-1] == 0:
            x[i] = ( random.random() < TransitionMatrix[0,1] ) * 1 #Proba of jump from 0 to 1
        else:
            x[i] = ( random.random() < TransitionMatrix[1,1] )*1 #proba of stay from 1 to 1
    return x


def add_temporal_edges_of_a_given_nodepair( temporal_edges, node1, node2, initialDistribution, TransitionMatrix ):
    T = len( temporal_edges )
    x = makeTimeSerie( T, initialDistribution, TransitionMatrix )
    
    if node1 < node2:
        edge = ( node1, node2 )
    else:
        edge = ( node1, node2 )
    
    for t in range( T ):
        if x[t] == 1:
            temporal_edges[ t ].append( edge )
    
    return temporal_edges


def makeTransitionMatrix( stationnaryDistribution, linkPersistence ):
    p = stationnaryDistribution[1]
    P = np.zeros( (2,2) )
    P[1,1] = linkPersistence
    P[1,0] = 1 - linkPersistence
    P[0,1] = p * ( 1-linkPersistence) / (1-p)
    P[0,0] = 1 - P[0,1]
    return P




def generateThetaDCSBM( N, law = 'sbm' ):
    if law == 'sbm':
        return np.ones( N )
    elif law == 'normal':
        theta = np.zeros( N )
        sigma = 0.25 #Here sigma for normal law can be changed
        for i in range( N ):
            #theta[ i ] = np.abs( 1 + np.random.normal( loc = 0, scale = 0.25 ) )
            theta[ i ] = np.abs( np.random.normal( loc = 0, scale = sigma ) ) + 1 - sigma * np.sqrt( 2 / np.pi )
            #TODO: seems the mean is not equal to 1 ?
        return theta

    elif law == 'pareto':
        a = 3 #Here can change the Pareto parameter
        return (np.random.pareto( a, N ) + 1) * (a-1) / a
    else:
        raise TypeError("The method is not implemented")