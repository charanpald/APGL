#A test to compare Python TreeRank versus R TreeRank 

library(TreeRank)

LRsvmLinear = function(formula, data, bestresponse, ...) {
  LRsvm(formula, data, bestresponse, kernel="vanilladot", C=10, ...)
} 

data("TRdata");
maxDepths = 3:10
numTrees = 5
i = 1

aucTrain = c()
aucTest = c()

for (maxDepth in maxDepths) { 
  #tree <- TreeRank(class~., Gauss2D.learn, growing=growing_ctrl(minsplit=50,maxdepth=maxDepth ,mincrit=0), bestresponse=1, LeafRank=LRCart)
  forest <- TreeRankForest(class~., Gauss2D.learn, growing=growing_ctrl(minsplit=50,maxdepth=maxDepth ,mincrit=0), bestresponse=1, LeafRank=LRsvmLinear, ntree=5, sampsize=1.0, varsplit=1.0)
  scoreTrain <- predict(forest,Gauss2D.learn)

  aucTrain[i] = auc(getROC(forest, Gauss2D.learn))
  aucTest[i] = auc(getROC(forest, Gauss2D.test))
  
  i = i+1
}
print(aucTrain)
print(aucTest)
