import numpy 
import multiprocessing 
import itertools
import scipy.sparse.linalg
from exp.util.GeneralLinearOperator import GeneralLinearOperator

def dot(args): 
    X, w = args 
    return X.dot(w)
    
def dot2(data, indices, indptr, shape, rowInds, i, W, queue): 
    data = numpy.array(data)
    indices = numpy.array(indices)
    indptr = numpy.array(indptr)
    Xr = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
    queue.put((i, Xr[rowInds[i]:rowInds[i+1], :].dot(W)))

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
        if not scipy.sparse.issparse(X): 
            raise ValueError("X matrix should be sparse") 
        if X.shape[0] != U.shape[0] or X.shape[1] != V.shape[0]: 
            raise ValueError("X and U s V^T should have the same shape")
        
        def matvec(w): 
            return X.dot(w) + (U*s).dot(V.T.dot(w)) 
        
        def rmatvec(w): 
            return X.T.dot(w) + (V*s).dot(U.T.dot(w))
            
        def matmat(W): 
            return X.dot(W) + (U*s).dot(V.T.dot(W))  
            
        def rmatmat(W): 
            return X.T.dot(W) + (V*s).dot(U.T.dot(W))
        
        return GeneralLinearOperator(X.shape, matvec, rmatvec, matmat, rmatmat, dtype=X.dtype) 
    
    @staticmethod 
    def parallelSparseLowRankOp(X, U, s, V): 
        numProcesses = multiprocessing.cpu_count()
        colInds = numpy.array(numpy.linspace(0, X.shape[1], numProcesses+1), numpy.int)
        
        def matvec(w): 
            pool = multiprocessing.Pool(processes=numProcesses) 
            paramList = [] 
            for i in range(numProcesses): 
                paramList.append((X[:, colInds[i]:colInds[i+1]], U, s, V[colInds[i]:colInds[i+1], :], w[colInds[i]:colInds[i+1]]))

            iterator = pool.imap(dotSVD, paramList, chunksize=1)

            #iterator = itertools.imap(dotSVD, paramList)
            p = numpy.zeros(X.shape[0])
            
            for i in range(numProcesses): 
                p += iterator.next()
                
            pool.terminate()
            
            return p
        
        def rmatvec(w):
            pool = multiprocessing.Pool(processes=numProcesses) 
            paramList = [] 
            for i in range(numProcesses): 
                paramList.append((X[:, colInds[i]:colInds[i+1]], U, s, V[colInds[i]:colInds[i+1], :], w))
        
            iterator = pool.imap(dotSVDT, paramList, chunksize=1)
            
            #iterator = itertools.imap(dotSVDT, paramList)
            p = numpy.zeros(X.shape[1])
            
            for i in range(numProcesses): 
                p[colInds[i]:colInds[i+1]] = iterator.next()
            
            pool.terminate()            
            
            return p     
            
        def matmat(W): 
            pool = multiprocessing.Pool(processes=numProcesses) 
            paramList = [] 
            for i in range(numProcesses): 
                paramList.append((X[:, colInds[i]:colInds[i+1]], U, s, V[colInds[i]:colInds[i+1], :], W[colInds[i]:colInds[i+1], :]))

            iterator = pool.imap(dotSVD, paramList, chunksize=1)

            #iterator = itertools.imap(dotSVD, paramList)
            P = numpy.zeros((X.shape[0], W.shape[1])) 
            
            for i in range(numProcesses):
                P += iterator.next()
                
            pool.terminate()
                
            return P
            
        def rmatmat(W): 
            pool = multiprocessing.Pool(processes=numProcesses) 
            paramList = [] 
            for i in range(numProcesses): 
                paramList.append((X[:, colInds[i]:colInds[i+1]], U, s, V[colInds[i]:colInds[i+1], :], W))

            iterator = pool.imap(dotSVDT, paramList, chunksize=1)

            #iterator = itertools.imap(dotSVD, paramList)
            P = numpy.zeros((X.shape[1], W.shape[1])) 
            
            for i in range(numProcesses): 
                P[colInds[i]:colInds[i+1], :] = iterator.next()
                
            pool.terminate()
            
            return P
            
        return GeneralLinearOperator(X.shape, matvec, rmatvec, matmat, rmatmat, dtype=X.dtype)     
        
        
    @staticmethod 
    def parallelSparseOp(X):
        """
        Return the parallel linear operator corresponding to left and right multiply of 
        csc_matrix X. Note that there is a significant overhead for creating and waiting 
        for locked processes. 
        """
        if not scipy.sparse.isspmatrix_csc(X): 
            raise ValueError("Currently only supports csc_matrices")
        
        #This doubles memory here but saves memory when on many CPUs and results in faster calculations when we do matmat 
        Xr = X.tocsr()
        numProcesses = multiprocessing.cpu_count()
        numJobs = numProcesses
        rowInds = numpy.array(numpy.linspace(0, X.shape[0], numJobs+1), numpy.int)
        colInds = numpy.array(numpy.linspace(0, X.shape[1], numJobs+1), numpy.int)
        
        XrData = multiprocessing.Array("d", Xr.data)
        XrIndices = multiprocessing.Array("i", Xr.indices)
        XrIndptr = multiprocessing.Array("i", Xr.indptr)
        
        def matvec(w): 
            pool = multiprocessing.Pool(processes=numProcesses) 
            paramList = [] 
            for i in range(numJobs): 
                paramList.append((Xr[rowInds[i]:rowInds[i+1], :], w))
            
            iterator = pool.imap(dot, paramList, chunksize=1)
            #iterator = itertools.imap(dot, paramList)
            p = numpy.zeros(X.shape[0])
            
            for i in range(numJobs): 
                p[rowInds[i]:rowInds[i+1]] = iterator.next()
            
            pool.terminate()
            return p
        
        def rmatvec(w): 
            pool = multiprocessing.Pool(processes=numProcesses) 
            paramList = [] 
            for i in range(numJobs): 
                paramList.append((X[:, colInds[i]:colInds[i+1]], w))
            
            iterator = pool.imap(dotT, paramList, chunksize=1)
            #iterator = itertools.imap(dotT, paramList)
            p = numpy.zeros(X.shape[1])
            
            for i in range(numJobs): 
                p[colInds[i]:colInds[i+1]] = iterator.next()
            
            pool.terminate()
            return p      
        
        def matmat(W): 
            queue = multiprocessing.Queue()
            workers = [] 
            
            for i in range(numJobs): 
                workers.append(multiprocessing.Process(target=dot2, args=(XrData, XrIndices, XrIndptr, X.shape, rowInds, i, W, queue)))
                
            P = numpy.zeros((X.shape[0], W.shape[1])) 
            
            for worker in workers:
                worker.start()
                
            for worker in workers:
                worker.join()
                
            while not queue.empty():
                i, result = queue.get()     
                P[rowInds[i]:rowInds[i+1], :] = result 
                
            for worker in workers:
                worker.terminate()
            
            return P    
            
        def rmatmat(W): 
            pool = multiprocessing.Pool(processes=numProcesses) 
            paramList = [] 
            for i in range(numJobs): 
                paramList.append((X[:, colInds[i]:colInds[i+1]], W))
            
            iterator = pool.imap(dotT, paramList, chunksize=1)
            #iterator = itertools.imap(dotT, paramList)
            P = numpy.zeros((X.shape[1], W.shape[1]))
            
            for i in range(numJobs): 
                P[colInds[i]:colInds[i+1], :] = iterator.next()
            
            pool.terminate()
            return P             
            
        return GeneralLinearOperator(X.shape, matvec, rmatvec, matmat, rmatmat, dtype=X.dtype)   
    