

import os
import sys
import errno
import logging
import numpy
import scipy.sparse
import matplotlib 
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt 
from apgl.graph import *
from exp.recommendexp.RecommendExpHelper import RecommendExpHelper
from exp.recommendexp.SyntheticDataset1 import SyntheticDataset1
from exp.recommendexp.CenterMatrixIterator import CenterMatrixIterator
from exp.util.SparseUtils import SparseUtils 
"""
Study the rank of the synthetic data and also the spectrum
""" 

generator = SyntheticDataset1(startM=5000, endM=10000, startN=1000, endN=1500, pnz=0.10, noise=0.01)
iterator = CenterMatrixIterator(generator.getTrainIteratorFunc()())

k = 1000


for i in range(5): 
    X = iterator.next()
    
    if i==0: 
        lastX = scipy.sparse.csc_matrix(X.shape)
    
    print("About to compute SVD") 
    U, s, V = SparseUtils.svdPropack(X, k) 
    print("Computed SVD") 
    
    plt.figure(0)
    plt.plot(numpy.arange(s.shape[0]), s) 
    
    deltaX = X - lastX 
    deltaX.eliminate_zeros()
    deltaX.prune()
    print(X.nnz-lastX.nnz)
    U, s, V = SparseUtils.svdPropack(deltaX, k) 
    
    plt.figure(1)
    plt.plot(numpy.arange(s.shape[0]), s) 
    
    lastX = X 
    
plt.show()
