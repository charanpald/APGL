

import numpy
import numpy.random as rand
from apgl.util.Parameter import Parameter 


class Latex(object):
    '''
    A class with some general useful static functions for Latex output.
    '''
    def __init__(self):
        pass


    @staticmethod
    def array1DToRow(X, precision=3):
        """
        Take a 1D numpy array and print in latex table row format i.e. x1 & x2 .. xn

        :param X: The array to print
        :type X: :class:`ndarray`

        :param precision: The precision of the printed floating point numbers.
        :type precision: :class:`int`
        """
        Parameter.checkInt(precision, 0, 10)
        if X.ndim != 1:
            raise ValueError("Array must be one dimensional")

        n = X.shape[0]
        outputStr = ""

        if X.dtype == float:
            fmtStr = "%." + str(precision) + "f & "
            endFmtStr = "%." + str(precision) + "f"
        else:
            fmtStr = "%d & "
            endFmtStr = "%d"

        for i in range(0, n):
            if i != n-1:
                outputStr += fmtStr % X[i]
            else:
                outputStr += endFmtStr % X[i]

        return outputStr


    @staticmethod
    def array2DToRows(X, Y=None, precision=3, bold=None, italic=None):
        """
        A method which will print line of a latex table using 2D arrays of X
        and Y. Prints in format X[0,0] (Y[0,0]) & X[0,1] (Y[0,1]) ...
        One can also supply the optional boolean matrices bold and italic which
        embolden or italicise the corresponding elements of X.

        :param X: The array to print
        :type X: :class:`ndarray`

        :param Y: The array to print in parantheses or None for no array.
        :type Y: :class:`ndarray`

        :param precision: The precision of the printed floating point numbers.
        :type precision: :class:`int`
        """
        if X.ndim != 2 or (Y!=None and Y.ndim != 2) :
            raise ValueError("Array must be two dimensional")
        if Y!=None and X.shape != Y.shape:
            raise ValueError("Arrays must be the same shape")

        n = X.shape[0]
        outputStr = ""

        if X.dtype == float:
            XfmtStr = "%." + str(precision) + "f"
        else:
            XfmtStr = "%d"

        if Y!= None:
            if Y.dtype == float:
                YfmtStr = "(%." + str(precision) + "f)"
            else:
                YfmtStr = "(%d)"

        for i in range(0, n):
            for j in range(X.shape[1]):
                tempXfmtStr = XfmtStr

                if bold != None and bold[i, j]:
                    tempXfmtStr = "\\textbf{" + tempXfmtStr + "}"

                if italic != None and italic[i, j]:
                    tempXfmtStr = "\\emph{" + tempXfmtStr + "}"

                if Y!=None:
                    fmtStr = tempXfmtStr + " " + YfmtStr
                    vals = (X[i, j], Y[i, j])
                else:
                    fmtStr = tempXfmtStr
                    vals = X[i, j]

                if j != X.shape[1]-1:
                    outputStr += fmtStr % vals + " & "
                else:
                    outputStr += fmtStr % vals

            outputStr += "\\\\"

            if i!=n-1:
                outputStr += "\n"

        return outputStr

    @staticmethod
    def listToRow(lst):
        """
        Take a list and convert into a row of a latex table.
        """
        Parameter.checkClass(lst, list)
        outputStr = ""

        for i in range(len(lst)):
            if i != len(lst)-1: 
                outputStr += str(lst[i]) + " & "
            else:
                outputStr += str(lst[i]) + "\\\\"

        return outputStr

    @staticmethod
    def addRowNames(lst, latexTable):
        """
        Take a list of names and add it to the string representing a Latex table
        latexTable. 
        """
        
        if latexTable.count("\\\\") != len(lst):
            raise ValueError("Number of items of list doesn't match table rows: " + str(len(lst)) + "!=" + str(latexTable.count("\\\\")))

        tableRows = latexTable.rstrip().rsplit("\n")
        newLatexTable = ""

        for i in range(len(tableRows)):
            row = tableRows[i]
            newLatexTable += lst[i] + " & " + row + "\n"

        return newLatexTable 
        
    @staticmethod 
    def latexTable(tableRows, caption="Insert caption here", header=False): 
        """
        Take a set of rows in Latex table format and wrap the text in a table 
        with a given caption. 
        """
        lineList = tableRows.splitlines()
        numRows = lineList[0].count("&") + 1 
                
        
        table = "\\begin{table}\n"
        table += "\\centering\n"
        table += "\\begin{tabular}{" + numRows*"l " + " }\n"
        table += "\\hline\n"
        
        count =0 
        for line in lineList: 
            table += line + "\n" 
            if count == 0 and header: 
                table += "\\hline\n"
            count += 1 
                            
            
        table += "\\hline\n" 
        table += "\\end{tabular}\n"
        table += "\\caption{" + caption + "}\n" 
        table += "\\end{table}\n"
        
        return table 
        