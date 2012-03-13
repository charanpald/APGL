from apgl.io.MultiGraphCsvReader import MultiGraphCsvReader
from apgl.util.PathDefaults import PathDefaults
from apgl.data.FeatureGenerator import FeatureGenerator
import datetime
import numpy
import logging

class CsvConverters():
    @staticmethod
    def provConv(x):
        provDict = {}
        provDict['CA'] = 0
        provDict['CF'] = 1
        provDict['CH'] = 2
        provDict['CM'] = 3
        provDict['GM'] = 4
        provDict['GT'] = 5
        provDict['HO'] = 6
        provDict['IJ'] = 7
        provDict['LH'] = 8
        provDict['LT'] = 9
        provDict['MT'] = 10
        provDict["PR"] = 11
        provDict['SC'] = 12
        provDict['SS'] = 13
        provDict['VC'] = 14

        return provDict[x]

    @staticmethod
    def detectionConv(x):
        detDict = {}
        detDict['Contactos VIH'] = 0  #Contact tracing
        detDict['Donantes'] = 1 #Blood donation
        detDict['Espontaneo Confidenci'] = 2  #Spontaneous
        detDict['ETS'] = 3 #STD
        detDict['Exterior'] = 4
        detDict['Flota'] = 5
        detDict['Gestantes'] = 6 #Pregnancy
        detDict["Internacionalistas"] = 7
        detDict['Ingresos'] = 8 #?
        detDict['Lacra'] = 9
        detDict['Poblacion'] = 10
        detDict['Reclusos'] = 11 #Prison population
        detDict['Trab. TURISMO'] = 12 #Tourist worker
        detDict['Hemofilicos'] = 13
        detDict['Trab. SALUD'] = 14
        detDict['Captados'] = 15 #Test 
        detDict['Nefropatas'] = 16
        detDict['Trab. CULTURA'] = 17 #Cultural worker
        detDict['Emigrantes'] = 18
        detDict['Tuberculosis'] = 19

        return detDict[x]

    @staticmethod
    def genderConv(x):
        genderDict = {'M': 0, 'F': 1}
        return genderDict[x]

    @staticmethod
    def orientConv(x):
        orientDict = {'HT': 0, 'HB': 1}
        return orientDict[x]

    @staticmethod
    def fteConv(x):
        fteDict = {}
        fteDict['"CAPTA"'] = 0
        fteDict['"CONTA"'] = 1
        fteDict['"CULTU"'] = 2
        fteDict['"DONAN"'] = 3
        fteDict['"EMIGR"'] = 4
        fteDict['"ESPON"'] = 5
        fteDict['"ETS"'] = 6
        fteDict['"EXTER"'] = 7
        fteDict['"FLOTA"'] = 8
        fteDict['"GESTA"'] = 9
        fteDict['"HEMOF"'] = 10
        fteDict['"INGRE"'] = 11
        fteDict['"LACRA"'] = 12
        fteDict['"NEFRO"'] = 13
        fteDict['"POBLA"'] = 14
        fteDict['"RECLU"'] = 15
        fteDict['"SALUD"'] = 16
        fteDict['"TUBER"'] = 17
        fteDict['"TURIS"'] = 18
        fteDict['"INTER"'] = 19
        return fteDict[x]

    @staticmethod
    def numContactsConv(x):
        """
        Convert to integer and return -1 if the value is missing.
        """
        
        try:
            return int(x)
        except ValueError:
            return float('nan')

    @staticmethod 
    def dateConv(x):
        """
        Conversion of a string formatted data d/m/y into the number of days
        since 1900. An error is raised for dates before 1900.
        """
        startDate = datetime.date(1900, 1, 1)

        try:
            inputDate = datetime.datetime.strptime(x, "%d/%m/%Y")
            inputDate = inputDate.date()
        except ValueError:
            #Return the current year 
            return float('nan')

        if inputDate.year < startDate.year:
            raise ValueError("Conversion of year before 1900 Error")

        tDelta = inputDate - startDate

        return tDelta.days

class HIVGraphReader():
    def __init(self):
        pass

    def readHIVGraph(self, undirected=True, indicators=True):
        """
        We will use pacdate5389.csv which contains the data of infection. The undirected
        parameter instructs whether to create an undirected graph. If indicators
        is true then categorical varibles are turned into collections of indicator
        ones. 
        """
        converters = {1: CsvConverters.dateConv, 3:CsvConverters.dateConv, 5:CsvConverters.detectionConv, 6:CsvConverters.provConv, 8: CsvConverters.dateConv }
        converters[9] = CsvConverters.genderConv
        converters[10] = CsvConverters.orientConv
        converters[11] = CsvConverters.numContactsConv
        converters[12] = CsvConverters.numContactsConv
        converters[13] = CsvConverters.numContactsConv

        def nanProcessor(X):
            means = numpy.zeros(X.shape[1])
            for i in range(X.shape[1]):
                if numpy.sum(numpy.isnan(X[:, i])) > 0:
                    logging.info("No. missing values in " + str(i) + "th column: " + str(numpy.sum(numpy.isnan(X[:, i]))))
                means[i] = numpy.mean(X[:, i][numpy.isnan(X[:, i]) == False])
                X[numpy.isnan(X[:, i]), i] = means[i]
            return X 

        idIndex = 0
        featureIndices = converters.keys()
        multiGraphCsvReader = MultiGraphCsvReader(idIndex, featureIndices, converters, nanProcessor)

        dataDir = PathDefaults.getDataDir()
        vertexFileName = dataDir + "HIV/alldata.csv"
        edgeFileNames = [dataDir + "HIV/grafdet2.csv", dataDir + "HIV/infect2.csv"]

        sparseMultiGraph = multiGraphCsvReader.readGraph(vertexFileName, edgeFileNames, undirected, delimiter="\t")

        #For learning purposes we will convert categorial variables into a set of
        #indicator features
        if indicators: 
            logging.info("Converting categorial features")
            vList = sparseMultiGraph.getVertexList()
            V = vList.getVertices(list(range(vList.getNumVertices())))
            catInds = [2, 3]
            generator = FeatureGenerator()
            V = generator.categoricalToIndicator(V, catInds)
            vList.replaceVertices(V)

        logging.info("Created " + str(sparseMultiGraph.getNumVertices()) + " examples with " + str(sparseMultiGraph.getVertexList().getNumFeatures()) + " features")

        return sparseMultiGraph

    def getIndicatorFeatureIndices(self):
        #TODO: Complete this function
        featureDict = {}
        featureDict["birthDate"] = 0
        featureDict["detectDate"] = 1


        featureDict["contactTrace"] = 2
        featureDict["donor"] = 3
        featureDict["randomTest"] = 4
        featureDict["STD"] = 5
        featureDict["prisoner"] = 13
        featureDict["recommendVisit"] = 17

        featureDict["CA"] = 22
        featureDict["CF"] = 23
        featureDict["CH"] = 24
        featureDict["CM"] = 25
        featureDict["GM"] = 26
        featureDict['GT'] = 27
        featureDict['HO'] = 28
        featureDict['IJ'] = 29
        featureDict['LH'] = 30
        featureDict['LT'] = 31
        featureDict['MT'] = 32
        featureDict["PR"] = 33
        featureDict['SC'] = 34
        featureDict['SS'] = 35
        featureDict['VC'] = 36

        featureDict["deathDate"] = 37
        featureDict["gender"] = 38
        featureDict["orient"] = 39

        featureDict["numContacts"] = 40
        featureDict["numTested"] = 41
        featureDict["numPositive"] = 42

        return featureDict

    def getNonIndicatorFeatureIndices(self):
        featureDict = {}
        featureDict["birthDate"] = 0
        featureDict["detectDate"] = 1
        featureDict["detectMethod"] = 2
        featureDict["province"] = 3
        featureDict["deathDate"] = 4
        featureDict["gender"] = 5
        featureDict["orient"] = 6

        return featureDict