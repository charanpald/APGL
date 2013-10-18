Graph Classes
=============

This package contains a number of graph types and corresponding utility classes. Graph structure (a set of edges) is stored using adjacency/weight matrices: DenseGraph uses numpy.ndarray, SparseGraph uses scipy.sparse and PySparseGraph uses pysparse. The advantage of using matrices is that one can store edges efficiently as sparse matrices and many graph algorithms can be computed efficiently and with relatively simple code. Furthermore, vertex labels are recorded in the VertexList and GeneralVertexList classes which store values as numpy.ndarrays and list elements respectively. However, the number of vertices remains fixed and one can only store non-zero floating point values on edges. 

In the case that one wishes to label vertices and edges with anything and efficiently add vertices, the DictGraph class can be used. To access the functionality of the other graph class classes it can be efficiently converted to one of the other graph types. 

.. toctree::
   :maxdepth: 1
   
   CsArrayGraph
   DenseGraph
   DictGraph
   DictTree 
   GeneralVertexList
   PySparseGraph
   SparseGraph
   VertexList 
