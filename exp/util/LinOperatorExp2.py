import multiprocessing 
import numpy 
import numpy.testing as nptst 
import scipy.sparse 
import time 
import gc 


def initProcess(XInput, WListInput):
    global X 
    global WList
    
    X = XInput 
    WList = WListInput     

def dot2(args):
    startTime = time.time()
    rowInds, i, q = args     
    
    global X
    global WList 
    
    W = WList[i]
    
    Y = X.dot(W)
    
    for j in range(q):
        Y = X.T.dot(Y)
        gc.collect() 
        Y = X.dot(Y)
        gc.collect()     
    
    print(time.time()-startTime)
    
    return Y

def parallelPowerIteration(X, W, q):
    """
    Take a sparse matrix X and compute (XX^T)^qX omega
    """
    numJobs = multiprocessing.cpu_count()
    colInds = numpy.array(numpy.linspace(0, X.shape[1], numJobs+1), numpy.int)

    #Store the data in X as RawArray objects so we can share it amoung processes
    XData = multiprocessing.RawArray("d", X.data)
    XIndices = multiprocessing.RawArray("i", X.indices)
    XIndptr = multiprocessing.RawArray("i", X.indptr)        
        
    #Split W beforehand 
    WList = []
    for i in range(numJobs): 
        WList.append(W[:, colInds[i]:colInds[i+1]])
    
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(), initializer=initProcess, initargs=(XData, XIndices, XIndptr, X.shape, WList)) 
    params = [] 
    
    for i in range(numJobs): 
        params.append((colInds, i, q))
    
    iterator = pool.map(dot2, params, chunksize=1)
    P = numpy.zeros((X.shape[0], W.shape[1])) 
    
    for i in range(numJobs): 
        P[:, colInds[i]:colInds[i+1]] = iterator[i]

    pool.close()
    pool.join()            
        
    return P   

def powerIteration(X, W, q): 
    Y = X.dot(W)
    
    for j in range(q):
        Y = X.T.dot(Y)
        gc.collect() 
        Y = X.dot(Y)
        gc.collect()  
        
    return Y 

if __name__ == '__main__':
    #Create a random sparse matrix X and a random dense one W     
    X = scipy.sparse.rand(10000, 8000, 0.2)
    X = X.tocsr()
    X.sort_indices()
    W = numpy.random.rand(8000, 200)
    
    q = 2    
    
    print("Starting multiplications")
    startTime = time.time()
    A = parallelPowerIteration(X, W, q)
    parallelTime = time.time()-startTime 
    
    startTime = time.time()
    B = powerIteration(X, W, q)
    nonParallelTime = time.time()-startTime 
        
    #nptst.assert_array_almost_equal(A, B)    
    
    print(parallelTime, nonParallelTime)