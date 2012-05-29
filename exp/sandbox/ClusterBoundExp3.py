"""
We test the cluster bound on the spectral clustering approach 
""" 
import numpy 

numpy.random.seed(21)
numpy.set_printoptions(suppress=True, precision=3)

nrows = 10 
AA = numpy.random.rand(nrows, 5)
AA = AA.dot(AA.T)

U = numpy.random.rand(nrows, 2)
U = U.dot(U.T)

k = 3 

lmbda, Q = numpy.linalg.eigh(AA)
sigma, P = numpy.linalg.eigh(U)

inds = numpy.flipud(numpy.argsort(lmbda)) 

indsk = inds[0:k]
Qk = Q[:, indsk] 
lmbdak = lmbda[indsk]

AAk = (Qk*lmbdak).dot(Qk.T)

#print(AAk - AA)
#Find component of U in space of A 
alpha = numpy.zeros(nrows) 
beta = numpy.zeros(nrows) 

R = (numpy.eye(nrows) - Q.dot(Q.T)).dot(P)
print(Q)
print(R)

for i in range(nrows): 
    #alpha[i] = lmbda[i] + Q[:, i].T.dot(U).dot(Q[:, i])
    alpha[i] = Q[:, i].T.dot(U).dot(Q[:, i])
    beta[i] = R[:, i].T.dot(U).dot(R[:, i])

print(lmbda)
print(alpha)
print(beta)
    
print((Q*alpha).dot(Q.T))  
print(U)

