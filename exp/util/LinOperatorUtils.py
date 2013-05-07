import numpy 
import multiprocessing 
import itertools
import scipy.sparse.linalg

def dot(args): 
    X, w = args 
    return X.dot(w)

def dotT(args): 
    X, w = args 
    return X.T.dot(w)

def dotSVD(args): 
    X, U, s, V, w = args 
    return X.dot(w) + (U*s).dot(V.T.dot(w)) 

def dotSVDT(args): 
    X, U, s, V, w = args 
    return X.T.dot(w) + (V*s).dot(U.T.dot(w))

class LinOperatorUtils(object): 
    """
    Some implementations of linear operators for various structured matrices. 
    """    
    def __init__(self): 
        pass 
    
    @staticmethod 
    def sparseLowRankOp(X, U, s, V): 
        def matvec(w): 
            return X.dot(w) + (U*s).dot(V.T.dot(w)) 
        
        def rmatvec(w): 
            return X.T.dot(w) + (V*s).dot(U.T.dot(w))
        
        return scipy.sparse.linalg.LinearOperator(X.shape, matvec, rmatvec, dtype=X.dtype) 
    
    @staticmethod 
    def parallelSparseLowRankOp(X, U, s, V): 
        numProcesses = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=numProcesses) 
        rowInds = numpy.array(numpy.linspace(0, X.shape[0], numProcesses+1), numpy.int) 
        colInds = numpy.array(numpy.linspace(0, X.shape[1], numProcesses+1), numpy.int)
        
        def matvec(w): 
            paramList = [] 
            for i in range(numProcesses): 
                paramList.append((X[rowInds[i]:rowInds[i+1], :], U[rowInds[i]:rowInds[i+1], :], s, V, w))
                
            iterator = pool.imap(dotSVD, paramList)
            #iterator = itertools.imap(dotSVD, paramList)
            p = numpy.zeros(X.shape[0])
            
            for i in range(numProcesses): 
                p[rowInds[i]:rowInds[i+1]] = iterator.next()
            
            return p
        
        def rmatvec(w):
            paramList = [] 
            for i in range(numProcesses): 
                paramList.append((X[:, colInds[i]:colInds[i+1]], U, s, V[colInds[i]:colInds[i+1], :], w))
                
            iterator = pool.imap(dotSVDT, paramList)
            p = numpy.zeros(X.shape[1])
            
            for i in range(numProcesses): 
                p[colInds[i]:colInds[i+1]] = iterator.next()
            
            return p      
            
        return scipy.sparse.linalg.LinearOperator(X.shape, matvec, rmatvec, dtype=X.dtype)   
        
        
    @staticmethod 
    def parallelSparseOp(X):
        numProcesses = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=numProcesses) 
        rowInds = numpy.array(numpy.linspace(0, X.shape[0], numProcesses+1), numpy.int) 
        colInds = numpy.array(numpy.linspace(0, X.shape[1], numProcesses+1), numpy.int)
        
        def matvec(w): 
            paramList = [] 
            for i in range(numProcesses): 
                paramList.append((X[rowInds[i]:rowInds[i+1], :], w))
                
            iterator = pool.imap(dot, paramList)
            p = numpy.zeros(X.shape[0])
            
            for i in range(numProcesses): 
                p[rowInds[i]:rowInds[i+1]] = iterator.next()
            
            return p
        
        def rmatvec(w): 
            paramList = [] 
            for i in range(numProcesses): 
                paramList.append((X[:, colInds[i]:colInds[i+1]], w))
                
            iterator = pool.imap(dotT, paramList)
            p = numpy.zeros(X.shape[1])
            
            for i in range(numProcesses): 
                p[colInds[i]:colInds[i+1]] = iterator.next()
            
            return p      
            
        return scipy.sparse.linalg.LinearOperator(X.shape, matvec, rmatvec, dtype=X.dtype)   
    