#source("Util.R")

#Do some model selection for the SVM parameter 
LRsvm2 = function(formula, data, bestresponse, ...) { 
  className = all.vars(formula)[1]
  yTrue = data[[className]]
  wpos = 1 - sum(yTrue==bestresponse)/length(yTrue)
  Cs = 2^(-2:6)
  print(c("Running model selection using Cs", Cs))
  
  folds = 3
  numExamples = nrow(data)
  sam <- stratifiedCV(yTrue, bestresponse, folds)
  if (is.numeric(sam) && sam == -1) { 
    print("Detected bad labels")
    ret <- list(label = names(sort(-table(yTrue)))[1], bestresponse = bestresponse)
    class(ret) <- "TR_LRsame" 
    return(ret) 
    }
  
  errors <- matrix(0, nrow = length(Cs), ncol = 1)
  i = 1 
  
  for (C in Cs) { 
    for (drop in 1:folds) {
      trainData = data[-sam[[drop]],]
      testData =  data[sam[[drop]],]
      
      if (length(unique(trainData[[className]])) < 2) { 
        print("Training data has only 1 label")
        } 
      
      svm = LRsvm(formula, trainData, bestresponse, C=C, ...)
      yPred = predict(svm, testData)
      yTrue = testData[[className]]
      errors[i] = errors[i] + bError(yPred, yTrue)
      } 
      i = i + 1
    }
    
  minInd = which.min(errors)
  bestC = Cs[minInd]
  print(errors)
  print(paste("Found best value of C as ", bestC))
  
  LRsvm(formula, data, bestresponse, C=bestC, wpos=wpos, ...); 
}

#Linear SVM without model selection
LRsvmLinearPlain <- function(formula, data, bestResponse, ...) { 
  LRsvm(formula, data, bestresponse, C=10, kernel="vanilladot", ...)
  }

#Basically the same as above but uses the linear kernel 
LRsvmLinear = function(formula, data, bestresponse, ...) { 
  className = all.vars(formula)[1]
  yTrue = data[[className]]
  wpos = 1 - sum(yTrue==bestresponse)/length(yTrue)
  Cs = 2^(-2:6)
  print(c("Running model selection using Cs", Cs))
  
  folds = 3
  numExamples = nrow(data)
  print(c("Number of examples ", numExamples))
  #Stratified cross validation 
  sam <- stratifiedCV(yTrue, bestresponse, folds)
  
  if (is.numeric(sam) && sam == -1){ 
    ret <- list(label = names(sort(-table(yTrue)))[1], bestresponse = bestresponse)
    class(ret) <- "TR_LRsame"
    return(ret) 
    }
  errors <- matrix(0, nrow = length(Cs), ncol = 1)
  i = 1 
  
  for (C in Cs) { 
    for (drop in 1:folds) {
      trainData = data[-sam[[drop]],]
      testData =  data[sam[[drop]],]
      
      if (length(unique(trainData[[className]])) < 2) { 
        print("Training data has only 1 label")
        } 
      
      svm = LRsvm(formula, trainData, bestresponse, C=C, kernel="vanilladot", ...)
      yPred = predict(svm, testData)
      yTrue = testData[[className]]
      
      errors[i] = errors[i] + bError(yPred, yTrue)
      } 
      i = i + 1
    }
    
  minInd = which.min(errors)
  bestC = Cs[minInd]
  print(errors)
  print(paste("Found best value of C as ", bestC))
  
  LRsvm(formula, data, bestresponse, C=bestC, wpos=wpos, kernel="vanilladot", ...); 
}


predict.TR_LRsame = function(object, newdata = NULL, ...) 
{
    rep(object$label, nrow(newdata))
}

LRCart2 = function(formula, data, bestresponse, ...) { 
  className = all.vars(formula)[1]
  yTrue = data[[className]]
  wpos = 1 - sum(yTrue==bestresponse)/length(yTrue)
  
  LRCart(formula, data, bestresponse, maxdepth = 10, minsplit = 30, wpos = wpos, nfcv = 3, ...);
}

LRforest2 = function(formula, data, bestresponse, ...) { 
  className = all.vars(formula)[1]
  yTrue = data[[className]]
  wpos = 1 - sum(yTrue==bestresponse)/length(yTrue)
  maxNodesList = 2^(1:3)
  print(c("Running model selection using maxNodesList", maxNodesList))
  
  folds = 3
  numExamples = nrow(data)
  print(c("Number of examples ", numExamples))
  #Stratified cross validation 
  sam <- stratifiedCV(yTrue, bestresponse, folds)
  
  if (is.numeric(sam) && sam == -1){ 
    ret <- list(label = names(sort(-table(yTrue)))[1], bestresponse = bestresponse)
    class(ret) <- "TR_LRsame"
    return(ret) 
    }
  
  errors <- matrix(0, nrow = length(maxNodesList), ncol = 1)
  i = 1 
  
  for (maxNodes in maxNodesList) { 
    for (drop in 1:folds) {
      print(c("Inner fold", drop))
      trainData = data[-sam[[drop]],]
      testData =  data[sam[[drop]],]
      
      if (length(unique(trainData[[className]])) < 2) { 
        print("Training data has only 1 label")
        } 
      
      forest = LRforest(formula, trainData, bestresponse, mtry=2*round(sqrt(ncol(data)-1)), prcsize = 0.5, wpos = wpos, ntree=100, maxnodes=maxNodes, ...);
      yPred = predict(forest, testData)
      yTrue = testData[[className]]
      errors[i] = errors[i] + bError(yPred, yTrue)
      } 
      i = i + 1
    }
    
  minInd = which.min(errors)
  bestMaxNodes = maxNodesList[minInd]
  print(errors)
  print(c("Found best value of maxNodes as ", bestMaxNodes))

  #In Breiman, 2001 he suggests using sqrt(numFeatures)
  LRforest(formula, data, bestresponse, mtry=2*round(sqrt(ncol(data)-1)), prcsize = 0.5, wpos = wpos, ntree=500, maxnodes=bestMaxNodes, ...);
}

filter = function(formula, data, className, N) { 
  classInd = match(className, names(data))
  exampleInds = attr(terms(formula, data=data), "term.labels")
  X = data[exampleInds]
  norms = colSums(X^2)
  inds = sort(norms, index.return=TRUE, decreasing=TRUE)$ix
  X = X[inds[1:N]]
  X = data.frame(X)
  
  form = as.formula(paste(className, "~", paste(names(X), collapse="+")))
  form
  }

functionalLR <- function(formula, data, bestresponse, leafRank, ...) { 
  className = all.vars(formula)[1]
  exampleInds = attr(terms(formula, data=data), "term.labels")
  yTrue = data[[className]]
  wpos = 1 - sum(yTrue==bestresponse)/length(yTrue)
  Ns = unique(round(seq(round(length(exampleInds)/40), round(length(exampleInds)/10), length.out=4))) 
  print(paste("Running model selection using Ns", paste(Ns, collapse=" ")))
  
  folds = 3
  numExamples = nrow(data)
  #Stratified cross validation 
  sam <- stratifiedCV(yTrue, bestresponse, folds)
  
  if (is.numeric(sam) && sam == -1) { 
    ret <- list(label = names(sort(-table(yTrue)))[1], bestresponse = bestresponse)
    class(ret) <- "TR_LRsame"
    return(ret) 
    }
  
  errors <- matrix(0, nrow = length(Ns), ncol = 1)
  i = 1 
  
  for (N in Ns) { 
    newFormula = filter(formula, data, className, N)
    
    for (drop in 1:folds) {
      trainData = data[-sam[[drop]],]
      testData =  data[sam[[drop]],]
      
      if (length(unique(trainData[[className]])) < 2) { 
        print("Training data has only 1 label")
        } 
      lr = leafRank(newFormula, trainData, bestresponse, ...)
      yPred = predict(lr, testData)
      yTrue = testData[[className]]
      errors[i] = errors[i] + bError(yPred, yTrue)
      } 
      i = i + 1
    }
    
  minInd = which.min(errors)
  bestN = Ns[minInd]
  print(c("Errors: ", errors)) 
  print(paste("Found best value of N as ", bestN))
  
  newFormula = filter(formula, data, className, bestN)
  leafRank(newFormula, data, bestresponse, ...)
}

LRCartF = function(formula, data, bestresponse, ...) {
  functionalLR(formula, data, bestresponse, LRCart2, ...)
}
  
LRsvmF = function(formula, data, bestresponse, ...) {
  functionalLR(formula, data, bestresponse, LRsvm2, ...)
}

