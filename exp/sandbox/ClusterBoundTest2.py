
from sympy import Symbol, simplify, collect, latex, pprint 
rho = Symbol("rho")
s = Symbol("s")
q = Symbol("q")
r = Symbol("r")
eps = Symbol("epsilon")
pprint(collect(simplify((q*(rho**2)*(s+1)*(rho-1)**2).expand() - 
    (2*q*rho*((rho-1)**2) * (rho*(s+1)-s) ).expand() + 
    (r*rho**2*(rho*(s+1)-s)**2).expand() - 
    (2*r*rho*(rho-1)*(rho*(s+1)-s)**2).expand() -
    (eps*((rho*(s+1)-s)**2) *(rho-1)**2).expand()), rho))