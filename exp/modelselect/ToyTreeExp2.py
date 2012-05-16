
"""
Let's generate a simple dataset using uniform distributions and see how the
decision tree works with the data. This time we look at binary labels. 
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy 
from exp.sandbox.predictors.PenaltyDecisionTree import PenaltyDecisionTree
from apgl.util.Evaluator import Evaluator 

numExamples = 1000 
numFeatures = 2 

numpy.random.seed(21)

X = numpy.random.rand(numExamples, numFeatures)
y = numpy.zeros(numExamples, numpy.int)
noise = 0.05 

for i in range(numExamples): 
    if X[i, 1] > 0.5: 
        if X[i, 0] > 0.6: 
            y[i] = 1
        else: 
            y[i] = -1
    else: 
        if X[i, 0] > 0.1: 
            y[i] = 1
        else: 
            if X[i, 1] < 0.1: 
                y[i] = 1
            else: 
                y[i] = -1 

X += numpy.random.randn(numExamples, numFeatures)*noise
y+= 1 

print(numpy.sum(y==2), numpy.sum(y==0)) 

trainSplit = 0.3
numTrainExamples = numExamples*trainSplit

trainX = X[0:numTrainExamples, :]
trainY = y[0:numTrainExamples]
testX = X[numTrainExamples:, :]
testY = y[numTrainExamples:]

learner = PenaltyDecisionTree(minSplit=1, maxDepth=50, pruning=False)
learner.learnModel(trainX, trainY)

predY = learner.predict(trainX)
print(Evaluator.binaryError(predY, trainY))
print(learner.getTree())


plt.figure(0)
plt.scatter(testX[:, 0], testX[:, 1], c=testY, s=50, vmin=0, vmax=2)
plt.title("Test set")
plt.colorbar()

plt.figure(1)
plt.scatter(trainX[:, 0], trainX[:, 1], c=trainY, s=50, vmin=0, vmax=2)
plt.title("Training set")
plt.colorbar()

colormap  = matplotlib.cm.get_cmap()

def displayTree(learner, vertexId, minX0, maxX0, minX1, maxX1, colormap): 
    vertex = learner.tree.getVertex(vertexId)
    if learner.tree.isLeaf(vertexId):
        p = mpatches.Rectangle([minX0, minX1], maxX0-minX0, maxX1-minX1, facecolor=colormap(vertex.getValue()/2.0), edgecolor="black")
        plt.gca().add_patch(p)            

    leftChildId = learner.getLeftChildId(vertexId)
        
    if learner.tree.vertexExists(leftChildId):
        if vertex.getFeatureInd() == 0: 
            displayTree(learner, leftChildId, minX0, vertex.getThreshold(), minX1, maxX1, colormap)
        else: 
            displayTree(learner, leftChildId, minX0, maxX0, minX1, vertex.getThreshold(), colormap)
        
    rightChildId = learner.getRightChildId(vertexId)    
    
    if learner.tree.vertexExists(rightChildId):
        if vertex.getFeatureInd() == 0: 
            displayTree(learner, rightChildId, vertex.getThreshold(), maxX0, minX1, maxX1, colormap)
        else: 
            displayTree(learner, rightChildId, minX0, maxX0, vertex.getThreshold(), maxX1, colormap)
            
plt.figure(2)
#p = mpatches.Rectangle([0, 0], 0.5, 0.5, facecolor=(0, 0, 1), edgecolor="red")
rootId = learner.tree.getRootId()
displayTree(learner, rootId, 0, 1, 0, 1, colormap)
plt.title("Tree default gamma")

numGammas = 10
gammas = numpy.linspace(0, 0.05, numGammas)
errors = numpy.zeros(numGammas)

learner = PenaltyDecisionTree(minSplit=10, maxDepth=5, pruning=True)
learner.setSampleSize(100)

for i in range(gammas.shape[0]): 
    print(gammas[i])
    learner.setGamma(gammas[i])
    learner.learnModel(trainX, trainY)
    predY = learner.predict(testX)
    errors[i] = Evaluator.binaryError(predY, testY)
    
plt.figure(3)
plt.scatter(gammas, errors)
plt.title("Error vs gamma")

#Now plot best tree 
plt.figure(4)
learner.setGamma(gammas[numpy.argmin(errors)])
learner.learnModel(trainX, trainY)
print(learner.gamma)

print(learner.tree)


rootId = learner.tree.getRootId()
displayTree(learner, rootId, 0, 1, 0, 1, colormap)
plt.title("Optimal tree")

plt.show()
    
#TODO: Compare with normal tree     
    