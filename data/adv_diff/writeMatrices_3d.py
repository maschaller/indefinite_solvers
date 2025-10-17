from fenics import *
from fenics_adjoint import *
from collections import OrderedDict
import scipy.sparse.linalg
import scipy.sparse
import scipy.io

import matplotlib.pyplot as plt

# Define a vector-valued expression
class MyVectorExpression(UserExpression):
    def eval(self, value, x):
        # Define the components of the vector
        value[0] = -0.5  # First component
        value[1] = 0 #x[0] - x[1]  # Second component
        value[2] = 0
    
    def value_shape(self):
        # Return the shape of the vector
        return (3,)

# Initialize the vector-valued expression
b = MyVectorExpression(degree=2)
c = Expression('1',degree = 1)
nu = Expression('1',degree = 0) #Expression('0.1*(1.5+sin(3*x[0]*x[1]))',degree = 2)

n = 81
# Next, we define the discretization space:

mesh = BoxMesh(Point(0,0,0),Point(1,1,1), n-1, n-1, n-1)
eta = FacetNormal(mesh)
f = Expression('10', degree=2)

V = FunctionSpace(mesh, 'P', 1)
    
# Define boundary condition
u_D = Constant(0) #Expression('0', degree=0)
def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, u_D, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

# - nu*inner(grad(u),eta)*v*ds\
F = ( nu*inner(grad(u), grad(v)))*dx\
    + c*u*v*dx\
    - inner(b,grad(u))*v*dx\
    - f*v*dx
a, L = lhs(F), rhs(F)

# Assemble matrix
A,load = assemble_system(a,L,bc)

a1PET = as_backend_type(A).mat()
ANP1 = scipy.sparse.csr_matrix(a1PET.getValuesCSR()[::-1], shape=a1PET.size)

tol = 0.5

#observation_region = Expression('x[0] >=  tol && x[1] >= tol && x[2] >= tol ? 1 : 0', degree=1,tol=tol)
#control_region = Expression('x[0] <=  tol && x[1] <= tol && x[2] <= tol ? 1 : 0', degree=1,tol=tol)     

observation_region = Expression('1', degree=1,tol=tol)
control_region = Expression('1', degree=1,tol=tol)     
 

mass = u*v*dx
b = control_region*u*v*dx
c = observation_region*u*v*dx 
 
B, dum = assemble_system(b,L,bc)
C, dum = assemble_system(c,L,bc)
Mass, dum = assemble_system(mass,L,bc)

#B = assemble(b)
#C = assemble(c)
#bc.apply(B)
#bc.apply(C)
#bc.apply(Mass)
 
BPET = as_backend_type(B).mat()
BNP = scipy.sparse.csr_matrix(BPET.getValuesCSR()[::-1], shape=BPET.size)

CPET = as_backend_type(C).mat()
CNP = scipy.sparse.csr_matrix(CPET.getValuesCSR()[::-1], shape=CPET.size)

MASSPET = as_backend_type(Mass).mat()
MassNP = scipy.sparse.csr_matrix(MASSPET.getValuesCSR()[::-1], shape=MASSPET.size)



mdic = {"A": ANP1, "b": load.get_local().T, "C": CNP, "B": BNP, "Mass": MassNP}
scipy.io.savemat('ad_fullobst_'+str(n)+'.mat', mdic)