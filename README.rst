Another Python Graph Library 
============================

This project develops a simple, fast and easy to use Python graph library using NumPy, Scipy and PySparse. For information on installing, and usage, see http://packages.python.org/apgl/index.html. 

Changelog 
---------
Changes in version 0.7.1: 

* Pickling of graphs 
* Update of ABCSMC algorithm for better multiprocessing 

Changes in version 0.7: 

* Python 3 support 
* Windows installer 
* DictTree - getRootId, children, leaves, nonLeaf, copy, pruning and subtree methods 
* Sampling returns tuple of ndarrays 
* Other minor changes and documentation improvements 

Changes in version 0.6.10: 

* Bootstrap and shuffle split sampling in Sampling class 
* Some refactoring for predictor wrappers 
* Fix for error with ErdosRenyi with DenseGraph
* Documentation improvements. 
* Fix for setting weight matrix for DenseGraph

Changes in version 0.6.9: 

* Erdos Renyi generator works with sparse graphs 
* SparseGraph Laplacian matrix methods return scipy.sparse matrices 
* Updated LibSVM wrapper 
* Other minor changes

Additional Information
----------------------
This library was written by Charanpal Dhanjal (email: charanpal at gmail dot com). If you find it useful, please cite APGL as follows: 

Charanpal Dhanjal, An Introduction to APGL, Statistics and Applications Group, Telecom ParisTech, 2012

