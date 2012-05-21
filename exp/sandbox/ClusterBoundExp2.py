
from sympy import Symbol, simplify, collect, latex, pprint, solve 


rho = Symbol("rho")
s = Symbol("s")
q = Symbol("q")
r = Symbol("r")
eps = Symbol("epsilon")
sigma = Symbol("sigma")


print(collect(simplify( ((q - sigma)**2*(sigma**2 * (s+1) - 2*q*sigma)).expand() 
    + r*sigma**2*s**2 - (2*r*s*sigma*(q-sigma)).expand() - ((q - sigma)**2*eps).expand()), sigma))

"""
print(solve(collect(simplify((q*(rho**2)*(s+1)*(rho-1)**2).expand() - 
    (2*q*rho*((rho-1)**2) * (rho*(s+1)-s) ).expand() + 
    (r*rho**2*(rho*(s+1)-s)**2).expand() - 
    (2*r*rho*(rho-1)*(rho*(s+1)-s)**2).expand() -
    (eps*((rho*(s+1)-s)**2) *(rho-1)**2).expand()), rho), rho))"""

"""pprint(collect(simplify((q*(rho**2)*(s+1)*(rho-1)**2).expand() - 
    (2*q*rho*((rho-1)**2) * (rho*(s+1)-s) ).expand() + 
    (r*rho**2*(rho*(s+1)-s)**2).expand() - 
    (2*r*rho*(rho-1)*(rho*(s+1)-s)**2).expand() -
    (eps*((rho*(s+1)-s)**2) *(rho-1)**2).expand()), rho))"""