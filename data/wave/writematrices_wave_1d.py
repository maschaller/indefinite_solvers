
from fenics import *
from dolfin import *

import scipy.sparse
import scipy.io


MeshN = 51

mesh = UnitIntervalMesh(MeshN)
n = FacetNormal(mesh)

hx = 1/MeshN

V_h = FiniteElement('CG', mesh.ufl_cell(), 1)
Q_h = FiniteElement('CG', mesh.ufl_cell(),  1)
W = FunctionSpace(mesh, V_h*Q_h)
V,Q = W.split()

V_collapse = V.collapse()

f = Expression('1', degree=1)

damping = Constant(1)
damping2 = Constant(1)
wavenumber = Constant(1)

(p, q) = TrialFunctions(W)
(v, w) = TestFunctions(W)

d = Function(V_collapse, name="data")

# Assemble A Part
F = (damping*p*v - wavenumber*q*v.dx(0))* dx  \
    + (damping2*q*w + wavenumber*p.dx(0)*w)* dx \
    + f*v*dx
#a = lhs(F), rhs(F)

a,L = lhs(F), rhs(F)

A = assemble(a)
load = assemble(L)

APET = as_backend_type(A).mat()
ANP1 = scipy.sparse.csr_matrix(APET.getValuesCSR()[::-1], shape=APET.size)

tol = 0
observation_region = Expression('x[0] >=  tol || x[1] >= tol ? 1 : 0', degree=0,tol=tol, d=d)
control_region = Expression('x[0] >=  tol || x[1] >= tol ? 1 : 0', degree=0,tol=tol, d=d)     
 
mass = (p*v + inner(q,w))* dx
b = (p*v + inner(q,w))*control_region*dx
c = (p*v + inner(q,w))*observation_region*dx 
 
B = assemble(b)
C = assemble(c)
Mass = assemble(mass)

BPET = as_backend_type(B).mat()
BNP = scipy.sparse.csr_matrix(BPET.getValuesCSR()[::-1], shape=BPET.size)

CPET = as_backend_type(C).mat()
CNP = scipy.sparse.csr_matrix(CPET.getValuesCSR()[::-1], shape=CPET.size)

MassPET = as_backend_type(Mass).mat()
MassNP = scipy.sparse.csr_matrix(MassPET.getValuesCSR()[::-1], shape=MassPET.size)



mdic = {"A": ANP1, "b": load.get_local().T, "C": CNP, "B": BNP, "Mass": MassNP}
scipy.io.savemat('cond/1d_'+str(MeshN)+'.mat', mdic)
