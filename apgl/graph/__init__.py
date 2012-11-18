from apgl.graph.SparseGraph import SparseGraph
from apgl.graph.GeneralVertexList import GeneralVertexList
from apgl.graph.DenseGraph import DenseGraph
from apgl.graph.DictGraph import DictGraph
from apgl.graph.VertexList import VertexList
from apgl.graph.GraphUtils import GraphUtils
from apgl.graph.GraphStatistics import GraphStatistics

#Optional modules are tried and ignored if not present 
try:
    from apgl.graph.PySparseGraph import PySparseGraph
except ImportError as error:
    pass

