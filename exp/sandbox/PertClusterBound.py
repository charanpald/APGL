"""
This is a cluster bound based on perturbation theory, Theorem 4.5 in the paper. 

"""
import numpy 

def frobeniusBound(omega, pi, k): 
    """
    Bound the canonical angles using Frobenius norm. 
    """
    normR = numpy.sqrt(omega[k:2k]**2).sum()
    delta = pi[k-1] - (pi[k] + omega[-1])
    
    return normR/delta 

def spectralBound(omega, pi, k): 
    """
    Bound the canonical angles using spectral norm. 
    """
    normR = omega[k]
    delta = pi[k-1] - (pi[k] + omega[-1])
    
    return normR/delta 

