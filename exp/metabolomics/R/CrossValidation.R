library(TreeRank)
library(ROCR)

#Some functions to do cross validation using TreeRank
outerCrossValidate <- function(XY) { 
  #Run outer cross validation to find model parameters 
  folds = 5 
  numExamples = nrow(XY)
  sam <- sample(1:numExamples)
  splitSize = ceiling(numExamples/folds)
  sam <- split(sam, rep(1:folds, splitSize)[1:numExamples])
  
  maxDepths = 3:5
  varsplits = seq(0.6, 1.0, 0.2)
  nfcvs = 1:2 
  mincrit = 0.01
  leafRanks = c(LRsvm2, LRCart2, LRforest)
  
  numParams = length(maxDepths)*length(varsplits)*length(nfcvs)*length(leafRanks)
  numParamTypes = 4 
  
  outerAUCs <- matrix(NA, nrow = folds, ncol = 2)
  bestParams = list()
  colnames(outerAUCs) <- c("Train","Test")

  for (drop in 1:folds) {
    trainData = XY[-sam[[drop]],]
    testData =  XY[sam[[drop]],]
    
    ind = 1
    bestAUC = 0 
    bestVarsplit = 0 
    bestMaxDepth = 0 
    
    aucs <- matrix(0, nrow = numParams, ncol = 2)
    colnames(aucs) <- c("Train","Test")
    
    for (maxDepth in maxDepths) { 
      for (varsplit in varsplits) { 
        for (nfcv in nfcvs) { 
          for (leafRank in leafRanks){ 
            innerAucs = crossValidate(trainData, maxDepth, varsplit, mincrit, nfcv, leafRank, folds)
            
            aucTrain = mean(innerAucs[,1]) 
            aucTest = mean(innerAucs[,2])
            
            aucs[ind,] = c(aucTrain, aucTest)
              
            if (aucs[ind, 2] > bestAUC) { 
              bestAUC = aucs[ind, 2]
              bestVarsplit = varsplit 
              bestMaxDepth = maxDepth 
              bestNfcv = nfcv
              bestLeafRank = leafRank  
              }
            
            ind = ind + 1 
            }
          }
        }
      }
      
      bestParamRow = list(bestMaxDepth, bestVarsplit, bestNfcv, bestLeafRank)
      bestParams = c(bestParams, bestParamRow)
  
      #Now pick smallest and train 
      growingControl = growing_ctrl(minsplit=50,maxdepth=bestMaxDepth,mincrit=mincrit)    
      treeCart <- TreeRank(formula=class~ .,data=trainData, bestresponse=1, varsplit=bestVarsplit, LeafRank=bestLeafRank, growing=growingControl, nfcv=bestNfcv)
    
      ROCTrain = getROC(treeCart, trainData)
      aucTrain = auc(ROCTrain)
      ROCTest = getROC(treeCart, testData)
      aucTest = auc(ROCTest)
          
      outerAUCs[drop,] <- c(aucTrain, aucTest)      
  }
  
  results = list(bestParams, outerAUCs)
  results  
}  
        
crossValidate <- function(XY, maxdepth, varsplit, mincrit, nfcv, leafRank, folds) { 
  numExamples = nrow(XY)
  sam <- sample(1:numExamples)
  splitSize = ceiling(numExamples/folds)
  sam <- split(sam, rep(1:folds, splitSize)[1:numExamples])  
  
  aucs <- matrix(NA, nrow = folds, ncol = 2)
  colnames(aucs) <- c("Train","Test")
  
  for (drop in 1:folds) {
      trainData = XY[-sam[[drop]],]
      testData =  XY[sam[[drop]],]
      
      growingControl = growing_ctrl(minsplit=50,maxdepth=maxdepth,mincrit=mincrit)
      startTime = Sys.time()
      treeCart <- TreeRank(formula=class~ .,data=trainData, bestresponse=1, varsplit=varsplit, LeafRank=leafRank, growing=growingControl, nfcv=nfcv)
      timeDiff = Sys.time() - startTime
      print(c("Time for TreeRank: ", timeDiff))
      print(c("Tree depth:", max(treeCart$depth)))
      print(c("Number of nodes:", length(treeCart$nodes) ))
      print(c("Parameters: ", maxdepth, varsplit, mincrit, nfcv))
      
      ROCTrain = getROC(treeCart, trainData) 
      aucTrain = auc(ROCTrain)
      ROCTest = getROC(treeCart, testData) 
      aucTest = auc(ROCTest)
      
      if(is.nan(aucTest)) { 
        aucTest = 0.5
        }
      
      print(c("AUC Train", aucTrain))
      print(c("AUC Test", aucTest))
        
      aucs[drop,] <- c(aucTrain, aucTest)
}
  aucs
}

saveResults <- function(X, D, className, savePrefix) { 
  Y <- D[, className]   
  XY <- cbind(X, Y)
  names(XY)[length(XY)]="class"
  XY = na.exclude(XY)
  
  print(c("Dimension of data:", dim(XY)))
  results = outerCrossValidate(XY)
  aucs = results[[2]]
  params = results[[1]]
  print(aucs)
  print(params)
  
  fileName = paste(savePrefix, className ,".dat", sep="") 
  save(aucs, params, file = fileName)
  }

