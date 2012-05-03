"""
read Bemol data and cluster users based on the number of similar products they purchased

Main function creates files storing a sub-part of the data
    -d dir, --dir dir:   data files are in directory dir
    -n N, --nb_user N:   number of users in the sub-dataset
    -D, --debug :        print debug message 
"""

from apgl.util.Util import Util
import math
import numpy
import logging
import sys
import os
import subprocess
import shutil
import gzip
import re
import getopt
from apgl.graph import *
from apgl.generator import *
from exp.sandbox.IterativeSpectralClustering import IterativeSpectralClustering
from exp.sandbox.GraphIterators import DatedPurchasesGraphListIterator
from exp.sandbox.GraphIterators import MyDictionary
from apgl.util.PathDefaults import PathDefaults

numpy.random.seed(21)
numpy.set_printoptions(suppress=True, precision=3, linewidth=200, threshold=40000)


class RGUsage(Exception):
    def __init__(self, msg):
        self.msg = msg

class RGIOError(Exception):
    def __init__(self, io_error, post_msg):
        self.io_error = io_error
        self.post_msg = post_msg
        
    def __str__(self):
        return str(self.io_error) + "\n" + self.post_msg

    @staticmethod
    def indent():
        return ' '*37

#======================================================
# this function should probably disappear
def cluster():
    k1 = 20 # numCluster to learn
    k2 = 40 # numEigenVector kept

    dir = PathDefaults.getDataDir() + "cluster/"
    graphIterator = getBemolGraphIterator(dir)
    #===========================================
    # cluster
    print "compute clusters"
    clusterer = IterativeSpectralClustering(k1, k2)
    clustersList = clusterer.clusterFromIterator(graphIterator, True)

    for i in range(len(clustersList)):
              clusters = clustersList[i]
              print(clusters)


#======================================================
#======================================================
# Bemol data manager
class BemolData:
    #===========================================
    @staticmethod
    def getGraphIterator(dir, nb_user=-1, nb_purchases_per_it=0):
        """
        * Load the Bemol data from the directory dir
        * The dataset will only contain the $nb_user$ first users. -1 means all users.
        * $nb_purchases_per_it$ is used to split the purchases list in several
        iterations. See DatedPurchasesGraphListIterator's documentation
        * Do not read all the data, only the $10^n$ version of data with $n$ big
        enough.
        """
        BemolData.assert_nb_user(nb_user)
        if nb_user == -1:
            nb_user = BemolData.nb_max_user()
            
        # read data
        f_data_name = BemolData.get_file_name(dir, nb_user)
        purchasesList = []
        dict_user = MyDictionary()
        try:
            f_data = gzip.open(f_data_name, 'rb')

            for line in f_data:
                m = re.match("(\d+)\s(\d+)\s(\d+)\s(\d+)", line)
                if dict_user.index(int(m.group(1))) < nb_user:
                    purchasesList.append([int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))])
            logging.info(" file read")

            # graph iterator
            graphIterator = DatedPurchasesGraphListIterator(purchasesList, nb_purchases_per_it)

            return graphIterator
        except IOError as error:
            raise RGIOError(error, RGIOError.indent() + 'consider running BemolData.generate_data_file(...)')

    #===========================================
    @staticmethod
    def generate_data_file(dir, nb_user=-1):
        logging.debug("nb_user: " + str(nb_user))
        BemolData.assert_nb_user(nb_user)
        if nb_user == -1:
            nb_user = BemolData.nb_max_user()

        # generate the file containing all the dataset
        # !!!!! security failure TOCTTOU
        f_data_name = BemolData.get_file_name(dir, BemolData.nb_max_user())
        if not os.path.exists(f_data_name):
            logging.info("creating file " + str(f_data_name))
            shutil.copy(BemolData.get_file_name(dir, -1), f_data_name)

        # other files to generate
        nb_user_to_generate = []
        current_nb_user = BemolData.get_nb_user_to_read(nb_user)
        logging.debug("current_nb_user before while: " + str(current_nb_user))
        # !!!!! security failure TOCTTOU
        while (not os.path.exists(BemolData.get_file_name(dir, current_nb_user))):
            logging.debug("current_nb_user in while: " + str(current_nb_user))
            nb_user_to_generate.append(current_nb_user)
            current_nb_user = BemolData.get_nb_user_to_read(current_nb_user+1)
        nb_user_to_generate.reverse()

    
        # generate other files
        for current_nb_user in nb_user_to_generate:
            # read data
            f_existing_data_name = BemolData.get_file_name(dir, current_nb_user+1)
            f_to_create_data_name = BemolData.get_file_name(dir, current_nb_user)
            logging.info("creating file " + f_to_create_data_name)
            dict_user = MyDictionary()
            try:
                f_existing_data = gzip.open(f_existing_data_name, 'rb')
                f_to_create_data = gzip.open(f_to_create_data_name, 'wb')

                i = 0
                i_max = BemolData.get_nb_line(f_existing_data_name)
                for line in f_existing_data:
                    Util.printIteration(i, 1000, i_max); i += 1
                    m = re.match("(\d+)\s(\d+)\s(\d+)\s(\d+)", line)
                    if dict_user.index(int(m.group(1))) < current_nb_user:
                        f_to_create_data.write(line)
            except IOError as error:
                if error.filename == f_existing_data:
                    raise RGIOError(error, RGIOError.indent() + 'it disappeared in the meanwhile')
                else:
                    raise error

    #===========================================
    # useful functions
    #===========================================

    #===========================================
    @staticmethod
    def nb_max_user():
        return 10**5

    #===========================================
    @staticmethod
    def assert_nb_user(nb_user):
        if nb_user > BemolData.nb_max_user():
            raise Exception("To much requested users. They are only " + str(BemolData.nb_max_user()))

    #===========================================
    @staticmethod
    def get_file_name(dir, nb_user=-1):
        BemolData.assert_nb_user(nb_user)
        if nb_user == -1:
            return dir +  "idu__idp__semaine__annee.dat.gz"
        else:
            return dir +  "idu__idp__semaine__annee__first_users" + str(BemolData.get_nb_user_to_read(nb_user)) + ".dat.gz"

    #===========================================
    @staticmethod
    def get_nb_user_to_read(nb_user=-1):
        BemolData.assert_nb_user(nb_user)
        if nb_user == -1:
            nb_user = BemolData.nb_max_user()
        # nb_user_to_read = 10^n or nb_max_user() 
        # with    10^n <= nb_user < 10^{n+1}
        nb_user_to_read = 10**math.ceil(math.log(nb_user, 10))
        if nb_user_to_read > BemolData.nb_max_user():
            nb_user_to_read = BemolData.nb_max_user()
        return int(nb_user_to_read)

    #===========================================
    @staticmethod
    def get_nb_line(f_name):
        return int(subprocess.check_output("zcat " + f_name + ' | wc -l | cut -d " " -f 1' , shell=True))


    #===========================================
    # main
    #===========================================

    #===========================================
    @staticmethod
    def main(argv=None):
        if argv is None:
            argv = sys.argv
        try:
            # read options
            try:
                opts, args = getopt.getopt(argv[1:], "hd:n:D", ["help", "dir=", "nb_user=", "debug"])
            except getopt.error, msg:
                 raise RGUsage(msg)
            # apply options
            dir = PathDefaults.getDataDir() + "cluster/"
            nb_user = -1
            log_level = logging.INFO
            for o, a in opts:
                if o in ("-h", "--help"):
                    print __doc__
                    return 0
                elif o in ("-d", "--dir"):
                    dir = a
                elif o in ("-n", "--nb_user"):
                    nb_user = int(a)
                elif o in ("-D", "--debug"):
                    log_level = logging.DEBUG
            logging.basicConfig(stream=sys.stdout, level=log_level, format='%(levelname)s (%(asctime)s):%(message)s')
            # process: generate data files
            BemolData.generate_data_file(dir, nb_user)
        except RGUsage, err:
            logging.error(err.msg)
            logging.error("for help use --help")
            return 2


#======================================================
#======================================================
if __name__ == "__main__":
    sys.exit(BemolData.main())

# to run
# python -c "execfile('exp/clusterexp/BemolData.py')" --help
# python2.7 -c "execfile('exp/clusterexp/BemolData.py')" --help

