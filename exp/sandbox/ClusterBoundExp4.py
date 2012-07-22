"""
This is a cluster bound based on perturbation theory. 

"""
import numpy 
from apgl.util import Util 
from apgl.graph import SparseGraph, GeneralVertexList 
from apgl.generator import ErdosRenyiGenerator 

numpy.random.seed(21)
numpy.set_printoptions(suppress=True, precision=3, linewidth=200, threshold=10000)

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

k = 45

lmbdaA, QA = numpy.linalg.eigh(AA)
lmbdaA, QA = Util.indEig(lmbdaA, QA, numpy.flipud(numpy.argsort(lmbdaA)))
lmbdaAk, QAk = Util.indEig(lmbdaA, QA, numpy.flipud(numpy.argsort(lmbdaA))[0:k])

lmbdaU, QU = numpy.linalg.eigh(U)
lmbdaU, QU = Util.indEig(lmbdaU, QU, numpy.flipud(numpy.argsort(lmbdaU)))

AAk = (QAk*lmbdaAk).dot(QAk.T)