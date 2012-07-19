"""
We test the cluster bound on the spectral clustering approach 
""" 
import numpy 
from apgl.util import Util 
from apgl.graph import SparseGraph, GeneralVertexList 
from apgl.generator import ErdosRenyiGenerator 

numpy.random.seed(21)
numpy.set_printoptions(suppress=True, precision=3, linewidth=200, threshold=10000)

def clusterBound(sigma, rho, lmbda, k): 
    """
    Compute the cluster bound corresponding to Theorem 4.1 of the paper. The 
    sigma array is eigenvalues of A, rho is eigenvalues of B, and lmbda is 
    the eigenvalues of (A_k + B). The rank of the required matrix is k. 
    """
    #First find gamma 
    gamma = sigma + rho[0] 
    for s in range(gamma.shape[0]): 
        for i in range(s+1): 
            for j in range(s-i+1): 
                val = sigma[i] + rho[j]
                if val < gamma[s]: 
                    gamma[s] = val
    
    #print("gamma=" + str(gamma))   
    #print("sigma=" + str(sigma))
    #print("rho=" + str(rho))
    
    
    r = (abs(rho)>10**-6).sum() 
    gammaSqSum = (gamma[0:k]**2).sum()
    lmbdaSqSum =  (lmbda[0:k]**2).sum()
    
    r2 = max((k-r), 0)
    sigmaSqSum = (sigma[0:r2]**2).sum()
    bound = gammaSqSum + lmbdaSqSum - 2*sigmaSqSum
    print("r=" + str(r))
    
    print("gammaSqSum=" + str(gammaSqSum))    
    print("lmbdaSqSum=" + str(lmbdaSqSum))    
    print("sigmaSqSum=" + str(sigmaSqSum))    
    
    return bound 

#Change to work with real Laplancian 
numRows = 100 
graph = SparseGraph(GeneralVertexList(numRows))

p = 0.1 
generator = ErdosRenyiGenerator(p)
graph = generator.generate(graph)

print(graph)

AA = graph.normalisedLaplacianSym()



p = 0.001
generator.setP(p)
graph = generator.generate(graph, requireEmpty=False)

AA2 = graph.normalisedLaplacianSym()


U = AA2 - AA

#print(U)

k = 45

lmbdaA, QA = numpy.linalg.eigh(AA)
lmbdaA, QA = Util.indEig(lmbdaA, QA, numpy.flipud(numpy.argsort(lmbdaA)))
lmbdaAk, QAk = Util.indEig(lmbdaA, QA, numpy.flipud(numpy.argsort(lmbdaA))[0:k])

lmbdaU, QU = numpy.linalg.eigh(U)
lmbdaU, QU = Util.indEig(lmbdaU, QU, numpy.flipud(numpy.argsort(lmbdaU)))



AAk = (QAk*lmbdaAk).dot(QAk.T)

lmbdaAU, QAU = numpy.linalg.eigh(AA + U)
lmbdaAU, QAU = Util.indEig(lmbdaAU, QAU, numpy.flipud(numpy.argsort(lmbdaAU)))
lmbdaAUk, QAUk = Util.indEig(lmbdaAU, QAU, numpy.flipud(numpy.argsort(lmbdaAU))[0:k])

lmbdaAkU, QAkU = numpy.linalg.eigh(AAk + U)
lmbdaAkU, QAkU = Util.indEig(lmbdaAkU, QAkU, numpy.flipud(numpy.argsort(lmbdaAkU)))
lmbdaAkUk, QAkUk = Util.indEig(lmbdaAkU, QAkU, numpy.flipud(numpy.argsort(lmbdaAkU))[0:k])

AUk = (QAUk*lmbdaAUk).dot(QAUk.T)
AkUk = (QAkUk*lmbdaAkUk).dot(QAkUk.T)

bound = clusterBound(lmbdaA, lmbdaU, lmbdaAkU, k)

print(lmbdaAUk)
print(lmbdaAkUk)

print(bound, numpy.linalg.norm(AUk - AkUk)**2) 

#print("realGamma=" + str(lmbdaAU))

print(numpy.trace(AUk.T.dot(AUk)))
print(numpy.trace(AkUk.T.dot(AkUk)))
print(numpy.trace(AkUk.T.dot(AUk)))

