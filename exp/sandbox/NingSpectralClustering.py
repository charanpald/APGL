"""
An implementation of the incremental spectral clustering method of Ning et al. given in
"Incremental Spectral Clustering by Efficiently Updating the Eigen-system".
"""
import time 
import numpy
import math
import logging
import scipy.linalg
from apgl.util.Util import Util
import scipy.cluster.vq as vq 

class NingSpectralClustering(object):
    def __init__(self, k):
        self.k = k
        self.kmeansIter = 20
        self.debugSave = False
        self.debugSVDiFile = 0

    def incrementEigenSystem(self, lmbda, Q, W, i, j, deltaW):
        """
        Updated an eigen system with eigenvalues lmbda and eigenvalues Q, with the
        change in incidence vector r. Also we have the weight matrix W. 
        """
        tol = 10**-3

        if self.debugSave:
            logging.warn("To debug we save current state. To be removed later on")
            numpy.save("lmbda", lmbda)
            numpy.save("Q", Q)
            numpy.save("W", W)
            numpy.save("i", i)
            numpy.save("j", j)
            numpy.save("deltaW", deltaW)



        n = W.shape[0]
        degrees = numpy.sum(W, 0)

        deltaDegrees = numpy.zeros(n)
        deltaDegrees[i] = deltaW
        deltaDegrees[j] = deltaW

        deltaL = numpy.zeros((n, n))
        deltaL[i, j] = -deltaW
        deltaL[j, i] = -deltaW
        deltaL[i, i] = deltaW
        deltaL[j, j] = deltaW

        newLmbda = lmbda.copy()
        newQ = Q.copy() 

        #Assume tau = 0
        largeNeighbours = numpy.union1d(numpy.nonzero(W[:, i])[0], numpy.nonzero(W[:, j])[0])
        largeNeighbours = numpy.union1d(largeNeighbours, [i,j])
        degreesHat = degrees[largeNeighbours]
        WHat = W[:, largeNeighbours]
        QHat = Q[largeNeighbours, :]

        #Estimate each eigenvector in turn
        for s in range(newLmbda.shape[0]):
            qi = Q[i, s]
            qj = Q[j, s]

            deltaDeltaQ = tol + 1
            deltaLmbda = 0 
            deltaQ = numpy.zeros(n)

            x = (qi - qj)
            y = (qi**2 + qj**2)
            a = x**2 - lmbda[s]*y
            c = deltaW*y

            iter = 0 
            while deltaDeltaQ > tol and iter < 2 :
                # --- updating deltaLmbda ---
                deltaQi = deltaQ[i]
                deltaQj = deltaQ[j]

                b = x*(deltaQi - deltaQj) - lmbda[s]*(qi*deltaQi + qj*deltaQj)
                
                d = numpy.sum(QHat[:, s]*degreesHat*deltaQ[largeNeighbours])
#                e = deltaW*(qi*deltaQi + qj*deltaQj)
                deltaLmbda = deltaW*(a+b)/(1+c+d)

                # --- updating deltaQ ---
                K = -WHat
                K[numpy.diag_indices(K.shape[1])] += (1-lmbda[s])*degreesHat
                h = (deltaLmbda*degrees + lmbda[s]*deltaDegrees)*Q[:, s]
                h -= deltaL.dot(Q[:, s])
                
                lastDeltaQ = deltaQ

                #Note that K can be singular 

                #Fix for weird error in pinv not converging
                try:
                    Hinv = scipy.linalg.pinv(K.T.dot(K))
                except scipy.linalg.linalg.LinAlgError as e:
                    # Least square didn't converge 
                    # so let's try SVD
                    logging.warn(str(e) + ". using pinv2 (based on SVD decomposition)")
                    try:
                        Hinv = scipy.linalg.pinv2(K.T.dot(K))
                    except scipy.linalg.linalg.LinAlgError as e:
                        # SVD didn't converge 
                        # so lets compute the pseudo inverse by ourself
                        logging.warn(str(e) + ". Computing pseudo inverse by ourself (using eigh)")
                        try:
                            localLmbda, localQ = scipy.linalg.eigh(K.T.dot(K))
                        except scipy.linalg.linalg.LinAlgError as e:
                            # eigh didn't work 
                            # so lets add a small term to the diagonal
                            logging.warn(str(e) + ". Adding a small diagonal term to obtain the eigen-decomposition")
                            alpha = 10**-5
                            localLmbda, localQ = scipy.linalg.eigh(K.T.dot(K)+ alpha*numpy.eye(K.shape[1]))
                            localLmbda -= alpha
                        nonZeroInds = numpy.nonzero(localLmbda)
                        localLmbda[nonZeroInds] = 1/localLmbda[nonZeroInds]
                        Hinv = localQ.dot(localLmbda).dot(localQ.T)
                    # to test different fixes and to submit a bug-report
                    self.debugSVDiFile += 1
                    numpy.savez("matrix_leading_to_pinv_error" + str(self.debugSVDiFile), K.T.dot(K))

                deltaQ[largeNeighbours] = Hinv.dot(K.T).dot(h)

                #Compute change in same way as paper? 
                deltaDeltaQ = scipy.linalg.norm(deltaQ[largeNeighbours] - lastDeltaQ[largeNeighbours])
                iter += 1 

            newLmbda[s] += deltaLmbda
            newDegrees = degrees + deltaDegrees

            newQ[:, s] += deltaQ
 
        pseudoScalarProduct = numpy.diag((newQ.T * newDegrees).dot(newQ))
        ind = numpy.nonzero(pseudoScalarProduct)[0]

        if ind.shape[0] < pseudoScalarProduct.shape[0]:
            logging.warn("Invalid eigenvector: removing ...")
            pseudoScalarProduct = pseudoScalarProduct[ind]
            newQ = newQ[:,ind]
            newLmbda = newLmbda[ind]

        newQ = newQ * pseudoScalarProduct**-0.5

        return newLmbda, newQ

    def __updateEigenSystem(self, lmbda, Q, deltaW, W):
        """
        Give the eigenvalues lmbda, eigenvectors Q and a deltaW matrix of weight
        changes, compute sequence of incidence vectors and update eigensystem.
        The deltaW is the change in edges from the current weight martrix which
        is given by W. 
        """
        n = deltaW.shape[0]
        lowTriInds = numpy.tril_indices(n)
        deltaW[lowTriInds] = 0
        changeInds = numpy.nonzero(deltaW)

        for s in range(changeInds[0].shape[0]):
            Util.printIteration(s, 10, changeInds[0].shape[0])
            i = changeInds[0][s]
            j = changeInds[1][s]

            assert deltaW[i, j] != 0
            if deltaW[i, j] < 0:
                logging.warn(" deltaW is usually positive (here deltaW=" +str(deltaW[i, j]) + ")")

            #Note: update W at each iteration here
            lmbda, Q = self.incrementEigenSystem(lmbda, Q, W, i, j, deltaW[i,j])
            W[i, j] += deltaW[i, j]
            W[j, i] += deltaW[i, j]
        
        return lmbda, Q 

    def cluster(self, graphIterator, T=10, timeIter=False):
        """
        Find a set of clusters using the graph and list of subgraph indices. The
        T parameter is how often one recomputes the eigenvalues.
        """
        tol = 10**-6 
        clustersList = []
        timeList = []

        iter = 0 

        for W in graphIterator:
            startTime = time.time()
            logging.info("Graph index:" + str(iter))

            if iter % T != 0:
                #Figure out the similarity changes in existing edges
                n = lastW.shape[0]
                deltaW = W.copy()
                deltaW[0:n, 0:n] = deltaW[0:n, 0:n] - lastW
                
                #If there are vertices added, add zero rows/cols to W
                WHat = lastW.copy()
                WHat = numpy.c_[numpy.r_[WHat, numpy.zeros((W.shape[0]-n, n))], numpy.zeros((W.shape[1], W.shape[0]-n))]
                
                Q = numpy.r_[Q, numpy.zeros((W.shape[0]-Q.shape[0], Q.shape[1]))]
                lmbda, Q = self.__updateEigenSystem(lmbda, Q, deltaW, WHat)
            else:
                logging.debug("Recomputing eigensystem")
                D = numpy.diag(numpy.sum(W, 0)) + tol*numpy.eye(W.shape[0])
                L = D - W
                lmbda, Q = scipy.linalg.eigh(L, D)
                lmbda = lmbda.real
                Q = Q.real
                inds = numpy.argsort(lmbda)[0:self.k]
                lmbda, Q = Util.indEig(lmbda, Q, inds)

            # Now do actual clustering 
            # V = vq.whiten(Q) is wrapped to care about lines with sd==0
            std_dev = numpy.std(Q, axis=0)
            std_dev[std_dev==0] = 1.
            V = Q / std_dev

            centroids, distortion = vq.kmeans(V, self.k, iter=self.kmeansIter)
            clusters, distortion = vq.vq(V, centroids)
            clustersList.append(clusters)
            timeList.append(time.time()-startTime)

            lastW = W.copy()
            iter += 1

        if timeIter:
            return clustersList, timeList
        else:
            return clustersList
