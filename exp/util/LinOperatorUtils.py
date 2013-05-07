import multiprocessing 
import scipy.sparse.linalg.LinearOperator

class LinOperatorUtils(object): 
    """
    Some implementations of linear operators for various structured matrices. 
    """    
        
    def __init__(self): 
        pass 
    
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
            
            for i in range(numProcesses): 
                
            
            return X.dot(v)
        
        def rmatvec(v): 
            return X.T.dot(v)          
            
        L = scipy.sparse.linalg.LinearOperator(X.shape, matvec, rmatvec) 
        return L    
    
    