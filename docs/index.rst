.. APGL documentation master file, created by
   sphinx-quickstart on Sat Apr 17 13:17:46 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to APGL's documentation!
================================

Another Python Graph Library is a simple, fast and easy to use graph library. The main characteristics are as follows: 

* Directed, undirected and multi-graphs using numpy and scipy matrices for fast linear algebra computations. The PySparseGraph and SparseGraph classes can scale up to 1,000,000s of vertices and edges on a standard PC. 
* Set operations including finding subgraphs, complements, unions, and intersections of graphs.
* Graph properties such as diameter, geodesic distance, degree distributions, eigenvector betweenness, and eigenvalues.  
* Other algorithms: search, Floyd-Warshall, Dijkstra's algorithm. 
* Configuration Model, Erdos-Renyi, Small-World, Albert-Barabasi and Kronecker graph generation 
* Write to Pajek and simple CSV files 
* Machine learning features - data preprocessing, kernels, PCA, KCCA, ABC, TreeRank.
* Unit tested using the Python unittest framework

Downloading
-----------
Download for Windows, Linux or Mac OS using: 

- Sourceforge `here <http://sourceforge.net/projects/apythongraphlib/>`_ 
- The Python Package Index (PyPI) `here <http://pypi.python.org/pypi/apgl/>`_ 

To use this library, you must have `Python <http://www.python.org/>`_, `NumPy <http://numpy.scipy.org/>`_ and `SciPy <http://www.scipy.org/>`_. The code has been verified on Python 2.7.2, Numpy 1.5.1 and Scipy 0.9.0, but should work with other versions. The automatic testing routine requires Python 2.7 or later, or the `unittest2 <http://pypi.python.org/pypi/unittest2/>`_ testing framework for Python 2.3-2.6 .

The source code repository is available `here <https://github.com/charanpald/APGL>`_ for those that want the bleeding edge, or are interested in development.  

Installation 
-------------
Ensure that `pip <http://pypi.python.org/pypi/pip>`_ is installed, and then install apgl in the following way: 

::

	pip install apgl

If installing from source unzip the apgl-x.y.z.tar.gz file and then run setup.py as follows: 

::

	python setup.py install 

In order to test the library (recommended), using the following commands in python 

::

	import apgl 
	apgl.test() 

and check that all tested pass. 


User Guide
----------

A short introduction to the main features of the library is available in the PDF document `"An Introduction to APGL" <ApglGuide.pdf>`_. This is the best way to learn the key features of APGL. In the meanwhile, here is small example of how to create a graph using the SparseGraph class which is based on scipy.sparse matrices. 

::

    >>> from apgl.graph import GeneralVertexList, SparseGraph 
    >>> import numpy 
    >>> numVertices = 5
    >>> graph = SparseGraph(GeneralVertexList(numVertices))
    >>> graph[0,1] = 1
    >>> graph[0,2] = 3
    >>> graph[1,2] = 0.1
    >>> graph[3,4] = 2
    >>> graph.setVertex(0, "abc") 
    >>> graph.setVertex(1, 123)
    >>> graph.findConnectedComponents()
    [[0, 1, 2], [3, 4]]
    >>> graph.getWeightMatrix()
    array([[ 0. ,  1. ,  3. ,  0. ,  0. ],
           [ 1. ,  0. ,  0.1,  0. ,  0. ],
           [ 3. ,  0.1,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  2. ],
           [ 0. ,  0. ,  0. ,  2. ,  0. ]])
    >>> graph.degreeDistribution()
    array([0, 2, 3])
    >>> graph.neighbours(0)
    array([2, 1], dtype=int32)
	>>> print(graph)
    SparseGraph: vertices 5, edges 4, undirected, vertex list type: GeneralVertexList

The :doc:`SparseGraph` is initialised as an undirected graph with :doc:`GeneralVertexList`, which stores the labels on vertices and can take any values as the vertex labels. Edges are added between vertices (0, 1), (0, 2), (1, 2) and (3, 4). Following, the first and second vertices (indexed by 0 and 1 respectively) are initialised with "abc" and 123 respectively, and we then compute some properties over the resulting graph. 

To learn more consult the reference documentation: 

.. toctree::
   :maxdepth: 1

   Reference


There is also a `PDF version  <AnotherPythonGraphLibrary.pdf>`_.
   
Support
--------

For any questions or comments please email me at <my first name> at gmail dot com. Documentation and code improvements/additions are especially welcome. The mailing list is accessible `here  
<https://lists.sourceforge.net/lists/listinfo/apythongraphlib-users>`_. 


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

