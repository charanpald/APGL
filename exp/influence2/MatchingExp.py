
"""
Looking at all articles with an abstract, restrict and save the experts 
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
    expertMatchesDict[field] = set([])    
    
    for expert in dataset.expertsDict[field]: 
        if expert in authorSet: 
            expertMatchesDict[field].add(expert)
            
    expertMatchesDict[field] = sorted(list(expertMatchesDict[field]))
    
#Now write out the matched experts 
for field in dataset.fields:
    outputFilename = dataset.getDataFieldDir(field) + "matched_experts.txt"
    lines = [x + "\n" for x in expertMatchesDict[field]]
    outputFile = open(outputFilename, "w")
    outputFile.writelines(lines)
    outputFile.close()
    
    logging.debug("Wrote experts to " + outputFilename)
