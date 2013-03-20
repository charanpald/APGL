def plotResults(method): 
    outputDir = PathDefaults.getOutputDir() + "erasm/" 
    errorFileName = outputDir + "results_" + method + ".npz"
    arr = numpy.load(errorFileName)      
    
    meanTrainErrors, meanTestErrors = arr["arr_0"], arr["arr_1"]
    
    plt.figure()
    plt.plot(ranks, meanTrainErrors, label="Train Error")
    plt.plot(ranks, meanTestErrors, label="Test Error")
    plt.xlabel("Rank")
    plt.ylabel("MSE")
    plt.legend()