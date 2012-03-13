ErdosRenyiGenerator
===================

A class to generate random graphs according to the Erdos-Renyi method. In this model, one specifies a probability p for the existence of an edge between two vertices. The following code generates a random graph with 10 vertices and edge probability 0.1. 

::
	
	import numpy 
	from apgl.graph import * 
	from apgl.generator.ErdosRenyiGenerator import ErdosRenyiGenerator
	
	p = 0.1
        graph = SparseGraph(VertexList(10, 1))
        generator = ErdosRenyiGenerator(p)
        graph = generator.generate(graph)

Methods 
-------
.. autoclass:: apgl.generator.ErdosRenyiGenerator.ErdosRenyiGenerator
   :members:
   :inherited-members:
   :undoc-members:
