
from sympy import Symbol, simplify, collect, latex, pprint, solve, S 


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
d = collect(simplify( ((-r - eps)*(1+t+s*y)**2).expand() + (1+s+t)*y**2*q**2 - (2*y*q**2*(1+t+s*y)).expand() + (c*(y**2 - 2*y)*(1+t+s*y)**2).expand()  ), y, evaluate=False)
print(collect(simplify( ((-r - eps)*(1+t+s*y)**2).expand() + (1+s+t)*y**2*q**2 - (2*y*q**2*(1+t+s*y)).expand() + (c*(y**2 - 2*y)*(1+t+s*y)**2).expand()  ), y))


print(d[y**4])
print(d[y**3])
print((2*c*s*(t+1-s)).expand())
print(d[y**2])
print((c*((t+1)*(t+1-4*s))).expand() - ((eps+r)*s**2).expand()  + (q**2*(1+t-s)).expand())
print(d[y])
print((-2*(q**2 + s*(r+eps) + c*(t+ 1))*(t+1)).expand())
print(d[S.One])
