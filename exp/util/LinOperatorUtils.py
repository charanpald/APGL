import numpy 
import multiprocessing 
import scipy.sparse.linalg.LinearOperator


def dot(args): 
    X, v = args 
    return X.dot(v)

def dotT(args): 
    X, v = args 
    return X.T.dot(v)

class LinOperatorUtils(object): 
    """
    Some implementations of linear operators for various structured matrices. 
    """    
        
    def __init__(self): 
        pass 
    
    @staticmethod 
    def sparseMatrixOp(X): 
        def matvec(v): 
            return X.dot(v)
        
        def rmatvec(v): 
            return X.T.dot(v)          
            
        L = scipy.sparse.linalg.LinearOperator(X.shape, matvec, rmatvec) 
        return L
        
    def parallelSparseMatrixOp(X): 
        def matvec(v): 
            numProcesses = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(processes=numProcesses) 
            paramList = [] 
            
            inds = numpy.array(numpy.linspace(0, X.shape[0], numProcesses+1))
            for i in range(numProcesses): 
                paramList.append((X[inds[i]:inds[i+1], :], v))
                
            iterator = pool.imap(dot, paramList)
            u = numpy.zeros(X.shape[0])
            
            for i in range(numProcesses): 
                u[inds[i]:inds[i+1]] = iterator.next()
            
            return u
        
        def rmatvec(v): 
            numProcesses = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(processes=numProcesses) 
            paramList = [] 
            
            inds = numpy.array(numpy.linspace(0, X.shape[0], numProcesses+1))
            for i in range(numProcesses): 
                paramList.append((X[:, inds[i]:inds[i+1]], v))
                
            iterator = pool.imap(dotT, paramList)
            u = numpy.zeros(X.shape[1])
            
            for i in range(numProcesses): 
                u[inds[i]:inds[i+1]] = iterator.next()
            
            return u      
            
        L = scipy.sparse.linalg.LinearOperator(X.shape, matvec, rmatvec) 
        return L    
    
    