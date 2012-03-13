library(RUnit)
source("UtilTest.R")

currentFile = "UtilTest.R"
testsuite.c2f <- defineTestSuite("c2f", dirs=c(getwd()), testFileRegexp = currentFile, testFuncRegexp = "^test.+") 
print(runTestSuite(testsuite.c2f))