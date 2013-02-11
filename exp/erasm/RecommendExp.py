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

try:
    import matplotlib.pylab as plb
except ImportError, exc:
    warn("Matplotlib must be installed to run Recommendations example.")


def readCoauthors(): 
    """
    Take a list of coauthors and read in the complete graph into a sparse 
    matrix R such that R_ij = k means author i has worked with j, k times.  
    """
    edgeFileName = PathDefaults.getOutputDir() + "edges.txt"    
        
    #First read the ids and figure out how big the matrix is 
    print("Starting to read edges")
    edges = numpy.loadtxt(edgeFileName, delimiter=",")
    
    print(edges.shape)
    

readCoauthors()