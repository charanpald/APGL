
from sympy import Symbol, simplify, collect, latex, pprint, solve 


rho = Symbol("rho")
s = Symbol("s")
q = Symbol("q")
r = Symbol("r")
eps = Symbol("epsilon")
y = Symbol("y")


"""print(collect(simplify( ((q - sigma)**2 * ((s+1)*sigma**2 - 2*q*sigma)).expand() 
    + r* s**2 * sigma**2 - (2*r*s*sigma*(q-sigma)).expand() - (eps*(q - sigma)**2).expand()), sigma))

"""
print(collect(simplify( (q**2*y**2*(s+1)).expand() - (2*y*q**2*(y+s)).expand() + (r*y**2*(y+s)**2).expand() - (2*r*y*(y+s)**2).expand() - (eps*(y+s)**2).expand()   ), y))
  

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
    
    #-epsilon*q**2 + sigma**4*(s + 1) + sigma**3*(-2*q*s - 4*q) + sigma**2*(-epsilon + q**2*s + 5*q**2 + r*s**2 + 2*r*s) + sigma*(2*epsilon*q - 2*q**3 - 2*q*r*s)