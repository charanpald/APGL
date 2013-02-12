"""
Test some recommendation with the movielen data 
"""

from apgl.util.PathDefaults import PathDefaults 
import nimfa
import numpy 
import numpy as np
import scipy.sparse as sp
from os.path import dirname, abspath, sep
from warnings import warn
import scipy.io 
import logging 
import sys
from sklearn.decomposition import ProjectedGradientNMF

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(linewidth=1000, threshold=10000000)

def preprocess(V):
    """
    Preprocess MovieLens data matrix. Normalize data.
    
    Return preprocessed target sparse data matrix in CSR format and users' maximum ratings. Returned matrix's shape is 943 (users) x 1682 (movies). 
    The sparse data matrix is converted to CSR format for fast arithmetic and matrix vector operations. 
    
    :param V: The MovieLens data matrix. 
    :type V: `scipy.sparse.lil_matrix`
    """
    print "Preprocessing data matrix ..."
    V = V.tocsr()
    maxs = [np.max(V[i, :].todense()) for i in xrange(V.shape[0])]
    now = 0
    for row in xrange(V.shape[0]):
        upto = V.indptr[row+1]
        while now < upto:
            col = V.indices[now]
            V.data[now] /= maxs[row]
            now += 1
    print "... Finished." 
    return V, maxs

def readCoauthors(): 
    """
    Take a list of coauthors and read in the complete graph into a sparse 
    matrix R such that R_ij = k means author i has worked with j, k times.  
    """
    matrixFileName = PathDefaults.getOutputDir() + "erasm/R"
    
    R = scipy.io.mmread(matrixFileName)
    logging.debug("Loaded matrix " + str(R.shape) + " with " + str(R.getnnz()) + " non zeros")   
    R = R.tocsr()
    #R = R[0:100 ,:]
    
    R, maxS = preprocess(R)
    print(R.getnnz())
    #print(R.todense())
    
    model = ProjectedGradientNMF(n_components=15, init='nndsvd', max_iter=100)    
    model.fit(R) 
    print("Done")
    
    W = model.components_
    print(W.shape)
    
    """
    model = nimfa.mf(R, 
                  seed = "random_vcol", 
                  rank = 12, 
                  method = "snmf", 
                  max_iter = 5, 
                  initialize_only = True,
                  version = 'r',
                  eta = 1.,
                  beta = 1e-4, 
                  i_conv = 10,
                  w_min_change = 0)
    print "Performing %s %s %d factorization ..." % (model, model.seed, model.rank) 
    fit = nimfa.mf_run(model)
    print "... Finished"
    sparse_w, sparse_h = fit.fit.sparseness()

    W = fit.basis()
    
    H = fit.coef()
       
    """       
       
readCoauthors()