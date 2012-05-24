
from sympy import Symbol, simplify, collect, latex, pprint, solve 


rho = Symbol("rho")
b = Symbol("b")
c = Symbol("c")
s = Symbol("s")
t = Symbol("t")
q = Symbol("q")
r = Symbol("r")
y = Symbol("y")
z = Symbol("z")
eps = Symbol("epsilon")
sigma = Symbol("sigma")


#print(collect(simplify( ((q - sigma)**2*(sigma**2 * (s+1) - 2*q*sigma)).expand() 
#    + r*sigma**2*s**2 - (2*r*s*sigma*(q-sigma)).expand() - ((q - sigma)**2*eps).expand()), sigma))


#Multi cluster bound 

print(collect(simplify( ((-r - eps)*(1+t+s*y)**2).expand() + (1+s+t)*y**2*q**2 - (2*y*q**2*(1+t+s*y)).expand() + (c*(y**2 - 2*y)*(1+t+s*y)**2).expand()  ), y))

