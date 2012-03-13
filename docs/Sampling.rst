Sampling
=================

This class can be used to evaluate a machine learning method with techniques such as cross validation and bootstrap. For example: 

::

	from apgl.util.Sampling import Sampling 
	idx = Sampling.crossValidation(3, 10) 
	for i,j in idx: 
		print(i, j)

	"""
	Outputs: 	
	([3, 4, 5, 6, 7, 8, 9], [0, 1, 2])
	([0, 1, 2, 6, 7, 8, 9], [3, 4, 5])
	([0, 1, 2, 3, 4, 5], [6, 7, 8, 9])
	"""

Methods 
-------
.. autoclass:: apgl.util.Sampling
   :members:
   :inherited-members:
   :undoc-members:
