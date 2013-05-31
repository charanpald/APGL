import numpy.testing as nptst 
import scipy.sparse
import math, sys, time
import pp
import numpy
import gc 
import time 

def initProcess(XInput, WListInput):
    global X 
    global WList
    
    X = XInput 
    WList = WListInput     

def dot2(i, q):
    startTime = time.time()    
    
    #global X
    WList = globals()["WList"]
    
    #print(X.shape)    
    
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
    ppservers = ()
    job_server = pp.Server(ppservers=ppservers)
    numJobs = job_server.get_ncpus()
    colInds = numpy.array(numpy.linspace(0, W.shape[1], numJobs+1), numpy.int)
    print(colInds)

    #Split W beforehand 
    WList = []
    for i in range(numJobs): 
        WList.append(W[:, colInds[i]:colInds[i+1]])
        
    initProcess(X, WList)
    
    global X2 
    X2 = 2
    #print(dir(globals()))
    
    jobList = []
    for i in range(numJobs): 
        jobList.append(job_server.submit(dot2, (i, q,), modules=("time", "gc"), globals=globals())) 
    
    P = numpy.zeros((X.shape[0], W.shape[1])) 
    
    for i in range(len(jobList)): 
        job = jobList[i]
        P[:, colInds[i]:colInds[i+1]] = job()         
        
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
    X = scipy.sparse.rand(1000, 800, 0.2)
    X = X.tocsr()
    X.sort_indices()
    W = numpy.random.randn(800, 200)
    
    q = 2    
    
    print("Starting multiplications")
    startTime = time.time()
    A = parallelPowerIteration(X, W, q)
    parallelTime = time.time()-startTime 
    
    startTime = time.time()
    B = powerIteration(X, W, q)
    nonParallelTime = time.time()-startTime 
        
    nptst.assert_array_almost_equal(A, B)    
    
    print(parallelTime, nonParallelTime)


