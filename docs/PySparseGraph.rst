PySparseGraph
=============

The PySparseGraph object represents a graph with an underlying sparse adjacency matrix representation implemented using PySparse. Therefore you must install Pysparse (http://pysparse.sourceforge.net/) in order to use this class. PySparseGraph has the advantage of being efficient in memory usage for graphs with few edges, and also relatively fast as Pysparse is implemented using C. Graphs of a 1,000,000 vertices or more can be created with minimal memory overheads. The following is a very simple example of how to use PySparseGraph:

::

	from apgl.graph import PySparseGraph 
	
	numVertices = 10

	graph = PySparseGraph(numVertices)
	graph[0, 2] = 1 
	graph[0, 3] = 1 
	graph[2, 1] = 1 
	graph[2, 5] = 1 
	graph[2, 6] = 1 
	graph[6, 9] = 1 
	
	subgraph = graph.subgraph([0,1,2,3])	

	graph.vlist[0] = "abc" 
	graph.vlist[1] =  123	

The code creates a new PySparseGraph with 10 vertices, after which edges are added and a subgraph is extracted using vertices 0, 1, 2, and 3. Notice that numpy.array vertices can be added to a PySparseGraph using the VertexList class in the constructor.  Finally, the first and second vertices are initialised with "abc" and 123 respectively
	

Methods 
-------
.. autoclass:: apgl.graph.PySparseGraph.PySparseGraph
   :members: 
   :inherited-members:
   :undoc-members:
   :show-inheritance:
