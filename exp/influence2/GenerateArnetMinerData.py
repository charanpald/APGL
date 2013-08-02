
from exp.influence2.ArnerMinetDataset import ArnerMinetDataset
import logging 
import sys 

"""
Create some graphs from the DBLP data. Basically, we use a seed list of experts 
and then find all the coauthors and their publications. 
"""

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

field = "Boosting" 
#field  = "IntelligentAgents"
#field  = "MachineLearning"
dataset = ArnerMinerDataset(field)
