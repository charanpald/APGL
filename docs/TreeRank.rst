TreeRank
========

TreeRank is a bipartite ranking algorithm which optimises the Receiver Operating Characteristic (ROC) curve. In a nutshell, the algorithm generates a ranking tree by recursive classification using weighted learners. The ranking tree can then be used to score a new test example according to which leaf that examples ends up at, and the score in turn is used to order/rank examples. The algorithm is described in detail in the article: S. Clemencon and N. Vayatis, Tree based ranking methods, IEEE Transactions on Information Theory, 2009.

An example of the usage of TreeRank is shown below: 

::

	import numpy 
	from apgl.metabolomics.TreeRank import TreeRank
	from apgl.metabolomics.leafrank.LinearSVM import LinearSVM
	from apgl.util.Evaluator import Evaluator 

	#Generate some random data
	numExamples = 200
	numFeatures = 10
	X = numpy.random.rand(numExamples, numFeatures)
	y = numpy.array(numpy.sign(numpy.random.rand(numExamples)-0.5), numpy.int)
	
	#Run TreeRank
	maxDepth = 2
	treeRank = TreeRank(LinearSVM.generate())
	treeRank.setMaxDepth(maxDepth)
	treeRank.learnModel(X, y)

	scores = treeRank.predict(X)
	#Print the AUC 
	print(Evaluator.auc(scores, y)) 
        
Here, we generate 200 random examples with 10 features, and associated binary -1/+1 labels in the vector y. TreeRank is run with a maximal depth of 2 and with a linear SVM as the "leafrank" algorithm (see the articles for more details). We then train using a call to learnModel and output the scores for the training set using predict. Finally, we output the Area Under the ROC Curve (AUC) using the predicted scores and the true labels. 

        

Methods 
-------
.. autoclass:: apgl.metabolomics.TreeRank.TreeRank
   :members:
   :inherited-members:
   :undoc-members:
