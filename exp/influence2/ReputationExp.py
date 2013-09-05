
"""
Find out which experts exist in the DBLP dataset and how many abstracts 
they have. 
"""
import logging 
import sys 
import itertools 
import numpy 
from exp.influence2.ArnetMinerDataset import ArnetMinerDataset

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

dataset = ArnetMinerDataset() 
#dataset.dataFilename = dataset.dataDir + "DBLP-citation-100000.txt" 

authorList, documentList, citationList = dataset.readAuthorsAndDocuments()
authorSet = set(itertools.chain.from_iterable(authorList))

print("Found all authors")
expertMatchesDict = {} 

for field in dataset.fields: 
    expertMatchesDict[field] = 0    
    
    for expert in dataset.expertsDict[field]: 
        if expert in authorSet: 
            expertMatchesDict[field] += 1
    
    expertMatchesDict[field] /= float(len(dataset.expertsDict[field]))

print(expertMatchesDict)
print(numpy.mean(expertMatchesDict.values())) 

