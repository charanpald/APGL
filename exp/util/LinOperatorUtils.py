import numpy 
import multiprocessing 
import itertools
import scipy.sparse.linalg
from exp.util.GeneralLinearOperator import GeneralLinearOperator

XrData = 0 
XrIndices = 0 
XrIntptr = 0 
XShape = 0 
WArray = 0 
WShape = 0

def initProcess(data, indices, indptr, shape, Warr, Wshp):
    global XrData 
    XrData = data 
    global XrIndices 
    XrIndices = indices 
    global XrIntptr 
    XrIntptr = indptr 
    global Xshape 
    Xshape = shape 
    
    global WArray
    WArray = Warr 
    global WShape 
    WShape = Wshp 

def dot(args): 
    X, w = args 
    return X.dot(w)
    
def dot2(args):
    rowInds, i = args     
    
    global XrData 
    global XrIndices
    global XrIntptr 
    global Xshape 

    data = numpy.frombuffer(XrData, dtype=numpy.float)
    indices = numpy.frombuffer(XrIndices, dtype=numpy.int32)
    indptr = numpy.frombuffer(XrIntptr, dtype=numpy.int32)
    Xr = scipy.sparse.csr_matrix((data, indices, indptr), shape=Xshape, copy=False)
    
    global WArray
    global WShape 
    W = numpy.frombuffer(WArray, dtype=numpy.float).reshape(WShape)

    return Xr[rowInds[i]:rowInds[i+1], :].dot(W)

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
    def sparseLowRankOp(X, U, s, V, parallel=False): 
        if X.shape[0] != U.shape[0] or X.shape[1] != V.shape[0]: 
            raise ValueError("X and U s V^T should have the same shape")
        
        if not parallel: 
            def matvec(w): 
                return X.dot(w) + (U*s).dot(V.T.dot(w)) 
            
            def rmatvec(w): 
                return X.T.dot(w) + (V*s).dot(U.T.dot(w))
                
            def matmat(W): 
                return X.dot(W) + (U*s).dot(V.T.dot(W))  
                
            def rmatmat(W): 
                return X.T.dot(W) + (V*s).dot(U.T.dot(W))
        else:
            def matvec(w): 
                return X.pdot(w) + (U*s).dot(V.T.dot(w)) 
            
            def rmatvec(w): 
                return X.T.pdot(w) + (V*s).dot(U.T.dot(w))
                
            def matmat(W): 
                return X.pdot(W) + (U*s).dot(V.T.dot(W))  
                
            def rmatmat(W): 
                return X.T.pdot(W) + (V*s).dot(U.T.dot(W))
        
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

            iterator = pool.map(dotSVD, paramList, chunksize=1)

            #iterator = itertools.imap(dotSVD, paramList)
            P = numpy.zeros((X.shape[0], W.shape[1])) 
            
            for i in range(numProcesses):
                P += iterator[i]
                
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
        
        XrData = multiprocessing.RawArray("d", numpy.array(Xr.data))
        XrIndices = multiprocessing.RawArray("i", Xr.indices)
        XrIndptr = multiprocessing.RawArray("i", Xr.indptr)

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
            WArray = multiprocessing.RawArray("d", W.flatten())
            pool = multiprocessing.Pool(processes=numProcesses, initializer=initProcess, initargs=(XrData, XrIndices, XrIndptr, X.shape, WArray, W.shape)) 
            params = [] 
            
            for i in range(numJobs): 
                params.append((rowInds, i))
            
            iterator = pool.map(dot2, params)
            P = numpy.zeros((X.shape[0], W.shape[1])) 
            
            for i in range(numJobs): 
                P[rowInds[i]:rowInds[i+1], :] = iterator[i]
                
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
    