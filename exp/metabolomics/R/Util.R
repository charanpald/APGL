# improved list of objects
.ls.objects <- function (pos = 1, pattern, order.by,
                        decreasing=FALSE, head=FALSE, n=5) {
    napply <- function(names, fn) sapply(names, function(x)
                                         fn(get(x, pos = pos)))
    names <- ls(pos = pos, pattern = pattern)
    obj.class <- napply(names, function(x) as.character(class(x))[1])
    obj.mode <- napply(names, mode)
    obj.type <- ifelse(is.na(obj.class), obj.mode, obj.class)
    obj.prettysize <- napply(names, function(x) {
                           capture.output(print(object.size(x), units = "auto")) })
    obj.size <- napply(names, object.size)
    obj.dim <- t(napply(names, function(x)
                        as.numeric(dim(x))[1:2]))
    vec <- is.na(obj.dim)[, 1] & (obj.type != "function")
    obj.dim[vec, 1] <- napply(names, length)[vec]
    out <- data.frame(obj.type, obj.size, obj.prettysize, obj.dim)
    names(out) <- c("Type", "Size", "PrettySize", "Rows", "Columns")
    if (!missing(order.by))
        out <- out[order(out[[order.by]], decreasing=decreasing), ]
    if (head)
        out <- head(out, n)
    out
}

# shorthand
lsos <- function(..., n=10) {
    .ls.objects(..., order.by="Size", decreasing=TRUE, head=TRUE, n=n)
}

#Generates some random data with the given number of examples and features
#The labels are also random and do not depend on the examples 
generateData <- function(numExamples=50, numFeatures=5) {
  X <- matrix(rnorm(numExamples* numFeatures), nrow = numExamples, ncol = numFeatures)
  Y =  sign(matrix(rnorm(numExamples, 1), nrow = numExamples, ncol = 1))     
  XY <- cbind(X, Y)
  XY = data.frame(XY)
  names(XY)[length(XY)]="class"
  
  XY
}

#Generates some random data with the given number of examples and features
#The labels are generated the relevant features and are  
generateData2 <- function(numExamples=50, numIrrFeatures=10, numRelFeatures=5) { 
    X <- matrix(rnorm(numExamples* numRelFeatures), nrow = numExamples, ncol = numRelFeatures)
    Xi <- matrix(rnorm(numExamples* numIrrFeatures), nrow = numExamples, ncol = numIrrFeatures)
    c <- matrix(rnorm(numRelFeatures), nrow=numRelFeatures, ncol=1)
    Y = X%*%c
    Y = Y - mean(Y)
    Y =  sign(Y) + 2 
    
    XY <- cbind(X*5, Xi, Y)
    XY = data.frame(XY)
    names(XY)[length(XY)]="class"
    
    XY
}

#Compute the error between two binary vectors     
bError <- function(yTrue, yPred) { 
  sum(yPred!=yTrue)
  }

stratifiedCV <- function(Y, bestresponse, folds) { 
  numExamples = length(Y)
  
  #Stratified cross validation 
  posYinds = (1:numExamples)[Y==bestresponse]
  negYinds = (1:numExamples)[Y!=bestresponse]
  posSplitSize = ceiling(length(posYinds)/folds)
  posSam <- split(sample(posYinds), rep(1:folds, posSplitSize)[1:length(posYinds)]) 
  negSplitSize = ceiling(length(negYinds)/folds)
  negSam <- split(sample(negYinds), rep(1:folds, negSplitSize)[1:length(negYinds)])
  
  print(paste("No. positive examples ", length(posYinds), " no. negative examples ", length(negYinds)))
  
  if (length(posYinds) < folds || length(negYinds) < folds) 
    sam = -1 
  else {
    sam = posSam
    
    for (i in 1:folds)
      sam[[i]] = c(sam[[i]], negSam[[i]])
    }
  sam 
  }
    
