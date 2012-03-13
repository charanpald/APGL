SmallWorldGenerator
====================

A class to generate random graphs according to the Small World method. The vertices are arranged on a two-dimensional lattice (a circle) and then edges are formed to their k closest neighbours in the lattice. Following, a random edge endpoint is selected and then rewired with probability p. The following code demonstrates the usage of this class. 

::
	
	import numpy 
	from apgl.graph import * 
	from apgl.generator.SmallWorldGenerator import SmallWorldGenerator
	
	k = 2
	p = 0.1
	graph = SparseGraph(VertexList(10, 1))
        generator = SmallWorldGenerator(p, k)
        graph = generator.generate(graph)

Methods 
-------
.. autoclass:: apgl.generator.SmallWorldGenerator.SmallWorldGenerator
   :members:
   :inherited-members:
   :undoc-members:
