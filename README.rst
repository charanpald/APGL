Another Python Graph Library 
============================

This project develops a simple, fast and easy to use Python graph library using NumPy, Scipy and PySparse. For information on installing, and usage, see http://packages.python.org/apgl/index.html. 

Changelog 
---------
Changes in version 0.7.3: 

* DictGraph - toSparseGraph, added depth and breadth first search, degree sequence, num directed edges, dijkstra's algorithm, adjacency list, and find all distances.
* Matrix graphs - toDictGraph, fixed depth and added breadth first searches 
* SparseGraph - toCsc, toCsr, specify sparse matrix format in constructor
* Other minor changes 

Changes in version 0.7.2: 

* Constructors for graph classes accept a size and weight matrix. 
* len property for vertex lists 
* Optimisation of depth first search, and ABCSMC 
* Lots of bug fixes and other improvements 

Changes in version 0.7.1: 

* Pickling of graphs 
* Update of ABCSMC algorithm for better multiprocessing 

Changes in version 0.7: 

* Python 3 support 
* Windows installer 
* DictTree - getRootId, children, leaves, nonLeaf, copy, pruning and subtree methods 
* Sampling returns tuple of ndarrays 
* Other minor changes and documentation improvements 


Additional Information
----------------------
This library was written by Charanpal Dhanjal (email: charanpal at gmail dot com). If you find it useful, please cite APGL as follows: 

Charanpal Dhanjal, An Introduction to APGL, Statistics and Applications Group, Telecom ParisTech, 2012

