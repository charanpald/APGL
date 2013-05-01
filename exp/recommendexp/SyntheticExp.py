
"""
Start with a simple toy dataset with time-varying characteristics 
"""

import numpy 
from exp.util.SparseUtils import SparseUtils 


startM = 5000 
startN = 10000 
endM = 6000
endN = 12000
r = 150 

U, s, V = SparseUtils.generateLowRank((endM, endN), r)

startNumInds = 8000
endNumInds = 10000
inds = numpy.random.randint(0, startM*startN-1, endNumInds)
inds = numpy.unique(inds)
numpy.random.shuffle(inds)
endNumInds = inds.shape[0]

matrixList = []

#In the first phase, the matrices stay the same size but there are more nonzero 
#entries 
print(endNumInds)

numMatrices = 10 
stepList = numpy.linspace(startNumInds, endNumInds, numMatrices)

for i in range(numMatrices): 
    X = SparseUtils.reconstructLowRank(U, s, V, inds[0:stepList[i]])
    X = X[0:startM, :][:, 0:startN]
    matrixList.append(X)
    
#Now we increase the size of matrix 
numMatrices = 10 
mStepList = numpy.linspace(startM, endM, numMatrices)
nStepList = numpy.linspace(startN, endN, numMatrices)

for i in range(numMatrices): 
    X = SparseUtils.reconstructLowRank(U, s, V, inds)
    X = X[0:mStepList[i], :][:, 0:nStepList[i]]
    matrixList.append(X)
    
for X in matrixList: 
    print(X.getnnz(), X.shape)
