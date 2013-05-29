import multiprocessing 
import numpy 
import numpy.testing as nptst 
import scipy.sparse 
import time 


def initProcess(data, indices, indptr, shape, Warr, Wshp):
    global XData 
    global XIndices 
    global XIntptr 
    global Xshape 
    
    XData = data 
    XIndices = indices 
    XIntptr = indptr 
    Xshape = shape 
    
    global WArray
    global WShape 
    
    WArray = Warr     
    WShape = Wshp 

def dot2(args):
    rowInds, i = args     
    
    global XData 
    global XIndices
    global XIntptr 
    global Xshape 

    data = numpy.frombuffer(XData, dtype=numpy.float)
    indices = numpy.frombuffer(XIndices, dtype=numpy.int32)
    indptr = numpy.frombuffer(XIntptr, dtype=numpy.int32)
    Xr = scipy.sparse.csr_matrix((data, indices, indptr), shape=Xshape)
    
    global WArray
    global WShape 
    W = numpy.frombuffer(WArray, dtype=numpy.float).reshape(WShape)

    return Xr[rowInds[i]:rowInds[i+1], :].dot(W)

def getMatmat(X): 
    numJobs = multiprocessing.cpu_count()
    rowInds = numpy.array(numpy.linspace(0, X.shape[0], numJobs+1), numpy.int)
    
    #Store the data in X as RawArray objects so we can share it amoung processes
    XData = multiprocessing.RawArray("d", X.data)
    XIndices = multiprocessing.RawArray("i", X.indices)
    XIndptr = multiprocessing.RawArray("i", X.indptr)
    
    def matmat(W): 
        WArray = multiprocessing.RawArray("d", W.flatten())
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(), initializer=initProcess, initargs=(XData, XIndices, XIndptr, X.shape, WArray, W.shape)) 
        params = [] 
        
        for i in range(numJobs): 
            params.append((rowInds, i))
        
        iterator = pool.map(dot2, params)
        P = numpy.zeros((X.shape[0], W.shape[1])) 
        
        for i in range(numJobs): 
            P[rowInds[i]:rowInds[i+1], :] = iterator[i]
            
        return P   
    
    return matmat 

if __name__ == '__main__':
    #Create a random sparse matrix X and a random dense one W     
    X = scipy.sparse.rand(10000, 8000, 0.1)
    X = X.tocsr()
    W = numpy.random.rand(8000, 200)
    
    startTime = time.time()
    A = getMatmat(X)(W)
    parallelTime = time.time()-startTime 
    
    startTime = time.time()
    B = X.dot(W)
    nonParallelTime = time.time()-startTime 
    
    nptst.assert_array_almost_equal(A, B)    
    
    print(parallelTime, nonParallelTime)