Parameter
=================
The Parameter class contains a set of static methods which are useful for type checking. For example

:: 

	from apgl.util.Parameter import Parameter 

	i = 5 
	j = 12
	min = 0 
	max = 10 
	
	#Parameter i checked as int and found to be within min and max 
	Parameter.checkInt(i, min, max) 
	
	#A ValueError is raised as j is greater than max 
	Parameter.checkInt(j, min, max) 


Methods 
-------
.. autoclass:: apgl.util.Parameter
   :members:
   :inherited-members:
   :undoc-members:
