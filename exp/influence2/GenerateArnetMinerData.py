
from exp.influence2.ArnetMinerDataset import ArnetMinerDataset
import logging 
import sys 
import numpy 

"""
Create some graphs from the DBLP data. Basically, we use a seed list of experts 
and then find all the coauthors and their publications. 
"""

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
numpy.set_printoptions(suppress=True, precision=3)

field = "Boosting" 
#field  = "IntelligentAgents"
#field  = "MachineLearning"
dataset = ArnetMinerDataset(field)
