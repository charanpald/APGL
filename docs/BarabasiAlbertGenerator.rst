BarabasiAlbertGenerator
=======================

A class to generate random graphs according to the Barabasi-Albert method. For example the following code generates a random graph with 10 vertices, starting with 2 initial vertices and adding 1 edge for each additional vertex. 

::

	from apgl.graph import * 
	from apgl.generator.BarabasiAlbertGenerator import BarabasiAlbertGenerator
	
	ell = 2
        m = 1
        graph = SparseGraph(VertexList(10, 1))
        generator = BarabasiAlbertGenerator(ell, m)
        graph = generator.generate(graph)

Methods 
-------
.. autoclass:: apgl.generator.BarabasiAlbertGenerator.BarabasiAlbertGenerator
   :members:
   :inherited-members:
   :undoc-members:
