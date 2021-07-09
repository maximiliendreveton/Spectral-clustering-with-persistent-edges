#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 14:47:08 2021


This is the code to plots the figure of the paper


@author: mdreveto
"""


import numpy as np
import random as random
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from tqdm import tqdm 
from itertools import permutations 
from itertools import combinations

#import os
#working_directory_path = os.getcwd() # Check current directory's path
#os.chdir(working_directory_path)
import preprocessing_high_school as high_school
import temporal_spectral_methods as temporal_spectral_method
import estimation_parameters as estimation_parameters

import MarkovBlockModel_edge_view as MBM


SIZE_TITLE = 24
SIZE_LABELS = 24
SIZE_TICKS = 18
SIZE_LEGEND = 18





"""
# =============================================================================
# CODE TO PLOT FIGURE 1
# =============================================================================

K = 3
N = 300
T = 30

muin = 0.04
muout = 0.02
( qin, qout ) = ( 0.9, 0.3 ) #Those are the intra and inter link persistence (number between 0 and 1, close to 1 means high link persistence across time, close to zero means spikes)

alpha = 1
betas = [ 0, 0.5, 1, 2, 5, 1000 ]
n_average = 2

Pin = MBM.makeTransitionMatrix( [ 1-muin, muin ], qin )
Pout = MBM.makeTransitionMatrix( [ 1-muout, muout ], qout )

#Careful: if K!= 3 the two lines below need to be changed accordingly
TransitionRateMatrix = np.array( [ [Pin, Pout, Pout], [Pout, Pin, Pout], [Pout, Pout, Pin] ] )
initialDistributionRateMatrix = np.array( [ [muin, muout, muout] , [muout, muin, muout], [muout, muout, muin] ] )

mean_accuracies = [  ]
std_accuracies = [ ]

accuracies = [ np.zeros( (n_average, T ) ) for i in range(len(betas)) ]

for trial in tqdm( range( n_average ) ):
    sizes =  list( np.random.multinomial( N, 1/K * np.ones( K ) ) )
    labels_true = []
    for i in range( K ):
        labels_true += [i+1] * sizes[i] 

    random.shuffle( labels_true )
    temporal_edges = MBM.makeMSBM_temporal_edges( N, T, initialDistributionRateMatrix, TransitionRateMatrix, labels_true, tqdm_ = False )

    for i in range( len( betas ) ):
        beta = betas[ i ]
        labels_pred_spectral = temporal_spectral_method.temporalSpectralClustering_knownParameters( N, temporal_edges, alpha, beta, K = K, useTqdm = False )
        accuracy_spectral = follow_accuracy_several_clusters( labels_true, labels_pred_spectral, K = K, useTqdm = False )
        accuracies[i][trial,:] = accuracy_spectral
    
for i in range( len( betas) ):
    mean_accuracies.append( np.mean( accuracies[ i ], axis = 0 ) )
    std_accuracies.append( np.std( accuracies[ i ], axis = 0 ) )

methods = betas
titleFig = 'Accuracy on MBM'
filename = 'MBM_N' + str(N) + '_K_' + str(K) + '_muin_' + str(muin) + '_muout_' + str(muout) + '_qin_' + str(qin) + '_qout_' + str(qout) + '_nAverage_' + str(n_average) + '.eps'
xticks = np.arange( 0, T, T // 5 )
plot_results( mean_accuracies, methods, xticks = xticks, std_accuracies = std_accuracies, titleFig = titleFig, saveFig = False, filename = filename, legend_title = 'beta' )




# =============================================================================
# CODE FOR FIGURE 2 (degree-corrected parameters)
# =============================================================================

law = 'normal'
K = 3
number_nodes_per_clusters = 100
T = 30
N = number_nodes_per_clusters * K

muin = 0.02
muout = 0.01
( qin, qout ) = ( 0.7, 0.4 ) #Those are the intra and inter link persistence (number between 0 and 1, close to 1 means high link persistence across time, close to zero means spikes)

alpha = 1
beta = 2
n_average = 25

Pin = MBM.makeTransitionMatrix( [ 1-muin, muin ], qin )
Pout = MBM.makeTransitionMatrix( [ 1-muout, muout ], qout )

TransitionRateMatrix = np.array( [ [Pin, Pout, Pout], [Pout, Pin, Pout], [Pout, Pout, Pin] ] )
initialDistributionRateMatrix = np.array( [ [muin, muout, muout] , [muout, muin, muout], [muout, muout, muin] ] )

mean_accuracies = [  ]
std_accuracies = [ ]

betas = [ 1, 3 ]

accuracies = [ np.zeros( ( n_average, T ) ) for i in range( len( betas ) ) ]

for trial in tqdm( range( n_average ) ):
    sizes = list( np.random.multinomial( N, 1/K * np.ones( K ) ) )
    labels_true = [ ]
    for i in range( K ):
        labels_true += [i+1] * sizes[i] 
    random.shuffle( labels_true )
    
    thetas = MBM.generateThetaDCSBM( N, law = law )
    temporal_edges = MBM.make_degree_corrected_SBM_Markov_temporal_edges( N, T, initialDistributionRateMatrix, TransitionRateMatrix, labels_true, thetas, tqdm_ = False )

    for i in range( len( betas ) ):
        beta = betas[ i ]
        labels_pred = temporal_spectral_method.temporalSpectralClustering_knownParameters( N, temporal_edges, alpha, beta, K = K, useTqdm = False )
        accuracy = follow_accuracy_several_clusters( labels_true, labels_pred, K = K, useTqdm = False )
        accuracies[i][trial,:] = accuracy
    
for i in range( len( betas ) ):
    mean_accuracies.append( np.mean( accuracies[ i ], axis = 0 ) )
    std_accuracies.append( np.std( accuracies[ i ], axis = 0 ) )


methods = betas
titleFig = 'Accuracy on MBM ' + str(law)
filename = law + '_MBM_' + str(N) + '_K_' + str(K) + '_muin_' + str(muin) + '_muout_' + str(muout) + '_qin_' + str(qin) + '_qout_' + str(qout) + '_nAverage_' + str(n_average) + '.eps'
xticks = np.arange( 0, T, T // 5 )
plot_results( mean_accuracies, methods, xticks = xticks, std_accuracies = std_accuracies, titleFig = titleFig, saveFig = False, filename = filename, legend_title = 'beta' )

"""





"""
# =============================================================================
# Plot results of the high school data set (Figure 3 of the paper)
# =============================================================================

year = 2013

if year == 2011:
    communities = [ ['PC'], ['PC*'], ['PSI*'] ] 
elif year == 2012:
    communities =  [ ['PC'], ['PC*'], ['MP*1'], ['MP*2'], ['PSI*'] ] #year 2012
elif year == 2013:
    communities = [ ['2BIO1'], ['2BIO2'], ['2BIO3'], ['MP*1'], ['MP*2'], ['MP'], ['PC'], ['PC*'], ['PSI*'] ]


if year == 2011 or year == 2012:
    labels_true, temporal_edges, days, node_indexing = high_school.preprocess_high_school_dataset_temporal_edges( groups_considered = communities, year = year )
elif year == 2013:
    labels_true, temporal_edges, days_timesteps, node_indexing = high_school.preprocess_high_school_dataset_temporal_edges( groups_considered = communities )


K = len( communities )
T = len( temporal_edges )
N = len( labels_true )


transitionsMatrix = np.zeros( (N,N,2,2), dtype=int )
for t in tqdm(range( 1, T)):
    transitionsMatrix = estimation_parameters.updateTransitionMatrix( temporal_edges[t-1], temporal_edges[t], transitionsMatrix )

nodesInEachCluster = []
for cluster in range( K ):
    nodesInEachCluster.append( [ i for i in range( N ) if labels_true[ i ] == cluster + 1 ] )


( Pin_theo, Pout_theo ) = estimation_parameters.parameterEstimation( transitionsMatrix, nodesInEachCluster, N , t = t, K = K )

l = estimation_parameters.log_likelihood_ratios( Pin_theo, Pout_theo )
alpha = l[0,1] + l[1,0] - 2*l[0,0]
beta = l[1,1] - l[0,0]



labels_pred_spectral_optimal = temporal_spectral_method.temporalSpectralClustering_knownParameters( N, temporal_edges, alpha, beta, K = K )
labels_pred_spectral_alpha_1_beta_1 = temporal_spectral_method.temporalSpectralClustering_knownParameters( N, temporal_edges, 1, 1, K = K )
labels_pred_spectral_alpha_2_9_beta_0_18 = temporal_spectral_method.temporalSpectralClustering_knownParameters( N, temporal_edges, 2.9, 0.18, K = K )


accuracy_spectral_optimal = follow_accuracy_several_clusters( labels_true, labels_pred_spectral_optimal, K = K, useTqdm= True, fast_accuracy_implementation = True )
accuracy_spectral_alpha_1_beta_1 = follow_accuracy_several_clusters( labels_true, labels_pred_spectral_alpha_1_beta_1, K = K, useTqdm= True, fast_accuracy_implementation = True )
accuracy_spectral_alpha_2_9_beta_0_18 = follow_accuracy_several_clusters( labels_true, labels_pred_spectral_alpha_2_9_beta_0_18, K = K, useTqdm= True, fast_accuracy_implementation = True )


if year == 2013:
    xticks = [0, 1500, 3000, 4500, 6000]
if year == 2012:
    xticks = [ 0, 2000, 4000, 6000, 8000, 10000 ]


methods = [ '(1,1)', '(2.9, 0.18)' ]
accuracies = [ accuracy_spectral_alpha_1_beta_1, accuracy_spectral_alpha_2_9_beta_0_18 ]
titleFig = 'Accuracy on High school database on year' + str( year )
filename = 'highschool_' + str(year) + '_aggregated_vs_optimal_from_2011.eps'
plot_results( accuracies, methods, xticks = xticks, titleFig = titleFig, saveFig = True, filename = filename, legend_title = '(alpha, beta)' )


"""





# =============================================================================
# Results for synthetic data
# =============================================================================





def plot_results( accuracies, methods, xticks, std_accuracies = [], titleFig = 'Title', saveFig = False , filename = 'Fig.eps', legend_title = 'Algorithm'):
    
    if std_accuracies == []:
        for i in range( len( accuracies ) ):
            plt.plot(accuracies[i], label = methods[i])
    else:
        for i in range( len( accuracies ) ):
            plt.errorbar( range(len(accuracies[i]) ), accuracies[ i ], yerr = std_accuracies[ i ], linestyle = '-.', label= methods[ i ] )

    legend = plt.legend( title=legend_title, loc=4,  fancybox=True, fontsize= SIZE_LEGEND )
    plt.setp( legend.get_title(),fontsize= SIZE_LEGEND )
    plt.xlabel( "Number of time steps", fontsize = SIZE_LABELS )
    plt.ylabel( "Accuracy", fontsize = SIZE_LABELS )
    plt.xticks( xticks, fontsize = SIZE_TICKS )
    plt.yticks( fontsize = SIZE_TICKS )
    if( saveFig ):
        plt.savefig( filename, format='eps', bbox_inches='tight' )
    else:
        plt.title( titleFig, fontsize = SIZE_TITLE )
    plt.show( )

    return 0




# =============================================================================
# FUNCTIONS RELATED TO CLUSTERING
# =============================================================================


def accuracy_several_clusters(labels_true, labels_pred, K = 2):
    accuracy = 0
    best_perm = []
    for perm in permutations( [ i + 1 for i in range(K) ] ): #permutations over the set {1,2,\dots, K }
        labels_pred_perm = [ perm[label-1] for label in labels_pred ] 
        if  accuracy_score(labels_true, labels_pred_perm ) > accuracy:
            accuracy = accuracy_score(labels_true, labels_pred_perm )
            best_perm = perm
    return accuracy, best_perm


def ideal_permutation_node_labels( node_labeling_1, node_labeling_2 ):
    labels_1 = set( node_labeling_1 )
    labels_2 = set( node_labeling_2 )
    
    labels = list( labels_1.union(  labels_2 ) )
    
    if len(node_labeling_1) != len(node_labeling_2):
        raise TypeError('The array have different sizes')
    
    best_perm = dict( )
    N = len( node_labeling_1 )
    
    partition_1 = [ [ i for i in range( N ) if node_labeling_1[ i ] == k  ] for k in labels ]
    partition_2 = [ [ i for i in range( N ) if node_labeling_2[ i ] == k  ] for k in labels ]
    
    partition_1 = sorted(partition_1, key=len, reverse=True)
    partition_2 = sorted(partition_2, key=len, reverse=True)
    
    sizes_1 = [ len( comm ) for comm in partition_1 ]
    sizes_2 = [ len( comm ) for comm in partition_2 ]
    
    node_labeling_2_permutated = np.zeros( N, dtype = int )
    
    empty_communities_index_2 = [ k for k in labels if len( [ i for i in range( N ) if node_labeling_2[ i ] == k  ] ) == 0 ]
    
    while partition_1:
        remaining_sizes_1 = sorted( list( set( [ len( comm ) for comm in partition_1 ] ) ), reverse = True)
        comm_1_of_largest_size = [ comm for comm in partition_1 if len( comm ) == remaining_sizes_1[0] ]
        for comm in comm_1_of_largest_size:
            best_overlap_with_comm, overlap = largest_overlap( comm, partition_2 )
            if overlap == 0 and len( empty_communities_index_2 ) > 0: 
                k = empty_communities_index_2[0]
                best_perm[ k ] = node_labeling_1[comm[ 0 ] ]
                empty_communities_index_2.remove( k )
            else :
                best_perm[ node_labeling_2[ best_overlap_with_comm[0] ] ] = node_labeling_1[comm[ 0 ] ]
            partition_1.remove( comm )
            partition_2.remove( best_overlap_with_comm )
        remaining_sizes_1.remove( remaining_sizes_1[0] )
    
    
    for i in range( N ):
        node_labeling_2_permutated[ i ] = best_perm[ node_labeling_2[i] ]
    
    return node_labeling_2_permutated, best_perm


def largest_overlap( community, communities ):
    overlap = [ ]
    for comm in communities:
        overlap.append( len( [node for node in community if node in comm ] ) )
    largest_overlap = np.argsort( overlap ) [-1]
    
    return communities[ largest_overlap ], sorted( overlap,  reverse = True)[0]


def follow_accuracy_several_clusters( labels_true, labelsPred, K = 2, useTqdm = True, fast_accuracy_implementation = False ):
    """
    This function compute the accuracy of the labelling labelsPred at each timestep t
    """
    accuracy = []
    if useTqdm:
        loop = tqdm( range(labelsPred.shape[1]) )
    else:
        loop = range(labelsPred.shape[1])
        
    for t in loop:
        if fast_accuracy_implementation:
            node_labeling, perm = ideal_permutation_node_labels( labels_true, list( labelsPred[:,t] ) )
            accuracy.append( accuracy_score( labels_true, node_labeling ) )
        else:
            acc, perm = accuracy_several_clusters(labels_true, labelsPred[:,t], K = K)
            accuracy.append( acc )
    return accuracy

