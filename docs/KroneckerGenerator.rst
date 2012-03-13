KroneckerGenerator
==================

A class to generate random graphs according to the Kronecker method. One specifies an input graph and the generated graph is grown using k successive kronecker multiplications of the adjacency matrix. 

::
	
	import numpy 
	from apgl.graph import * 
	from apgl.generator.KroneckerGenerator import KroneckerGenerator
	
	initialGraph = SparseGraph(VertexList(5, 1))
	initialGraph.addEdge(1, 2)
	initialGraph.addEdge(2, 3) 
	
	for i in range(5): 
		initialGraph.addEdge(i, i)
	
	k = 2
        generator = KroneckerGenerator(initialGraph, k)
        graph = generator.generate()

Methods 
-------
.. autoclass:: apgl.generator.KroneckerGenerator.KroneckerGenerator
   :members:
   :inherited-members:
   :undoc-members:
