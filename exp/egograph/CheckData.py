"""
We want to compare distributions of reading the csv file and the mat file
of transmissions. 
"""
from apgl.egograph.EgoUtils import EgoUtils
from apgl.data.ExamplesList import ExamplesList
import logging
from apgl.io.EgoCsvReader import EgoCsvReader
from apgl.util.Util import Util
import sys
import numpy

"""
Check the distributions of the features found in the transmissions data and in the
EgoData.csv file. We expect the means and variances to be similar.
"""
def checkDistributions():
    matFileName = "../../data/EgoAlterTransmissions.mat"
    examplesList = ExamplesList.readFromMatFile(matFileName)

    numFeatures = examplesList.getDataFieldSize("X", 1)
    X = examplesList.getDataField("X")[:, 0:numFeatures/2]
    Z = examplesList.getDataField("X")[:, numFeatures/2:numFeatures]
    y = examplesList.getDataField("y")
    A = Z[y==-1, :]

    #Now load directly from the CSV file
    #Learn the distribution of the egos
    eCsvReader = EgoCsvReader()
    egoFileName = "../../data/EgoData.csv"
    alterFileName = "../../data/AlterData.csv"
    egoQuestionIds = eCsvReader.getEgoQuestionIds()
    alterQuestionIds = eCsvReader.getAlterQuestionIds()
    (X2, titles) = eCsvReader.readFile(egoFileName, egoQuestionIds)
    X2[:, eCsvReader.ageIndex] = eCsvReader.ageToCategories(X2[:, eCsvReader.ageIndex])

    (mu, sigma) = Util.computeMeanVar(X)
    (mu2, sigma2) = Util.computeMeanVar(X2)
    (mu3, sigma3) = Util.computeMeanVar(Z)
    (mu4, sigma4) = Util.computeMeanVar(A)

    #Seems okay. Next check alters
    print(("Mean " + str(mu - mu4)))
    print(("Variance " + str(numpy.diag(sigma - sigma4))))

    """
    Analysis between the Egos in EgoData.csv and those in EgoAlterTransmissions.mat
    reveals that the distributions match closely. The main differences are
    in the means and variances in Q44A - D, but this isn't too suprising.
    """

    """
    In the alters case: 2, 12, 15, 18, 21-27, 44, (21-23 have very high variance)
    Checked that they are the correct fields as given in "Variable Pairs Jul29.xls"
    Checked that they are the right fields - Q50 > Q190$ (fixed)
    """

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    numpy.set_printoptions(precision=4, suppress=True)
    
    #checkDistributions()

    """
    We will read ego and alters data, and check they have the same values.
    """

    egoFileName = "../../data/EgoData.csv"
    alterFileName = "../../data/AlterData.csv"

    eCsvReader = EgoCsvReader()
    egoQuestionIds = eCsvReader.getEgoQuestionIds()
    alterQuestionIds = eCsvReader.getAlterQuestionIds()

    missing = 0 
    (egoX, titles) = eCsvReader.readFile(egoFileName, egoQuestionIds, missing)
    egoX[:, eCsvReader.ageIndex] = eCsvReader.ageToCategories(egoX[:, eCsvReader.ageIndex])

    (alterX, titles) = eCsvReader.readFile(alterFileName, alterQuestionIds, missing)
    alterX[:, eCsvReader.ageIndex] = eCsvReader.ageToCategories(alterX[:, eCsvReader.ageIndex])

    numFeatures = egoX.shape[1]
    numEgoExamples = egoX.shape[0]
    numAlterExamples = alterX.shape[0]

    for i in range(0, numFeatures):
        (histE, uniqElementsE) = Util.histogram(egoX[:, i])
        (histA, uniqElementsA) = Util.histogram(alterX[:, i])

        print((str(i) + " " + str(egoQuestionIds[i])))
        print(("Ego   " + str(uniqElementsE)))
        print(("Alter " + str(uniqElementsA)))
        print((numpy.setxor1d(uniqElementsE, uniqElementsA)))
        print((histE/numEgoExamples))
        print((histA/numAlterExamples))

    """
    Conclusion is that the distributions are broadly the same. The problem occurs
    with missing data handling. For example in Ego there are values with [ 0.  8.]
    with most zero, and in alter [ 0.  5.]. The means will be approx 8 for ego and 5 for
    alter.
    """