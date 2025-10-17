from fenics import *
from fenics_adjoint import *
from collections import OrderedDict
import scipy.sparse.linalg
import scipy.sparse
import scipy.io

# Define a vector-valued expression
class MyVectorExpression(UserExpression):
    def eval(self, value, x):
        value[0] = 1
    
    def value_shape(self):
        return (1,)

# Initialize the vector-valued expression
b = MyVectorExpression(degree=2)
c = Expression("0",degree = 1)

nu = Constant(1) # diffusivity

n = 21

# Next, we define the discretization space:
mesh = UnitIntervalMesh(n-1)
f = Expression('sin(x[0])', degree=2)
V = FunctionSpace(mesh, 'P', 1)

    
# Define boundary condition
u_D = Constant(0) #Expression('0', degree=0)
def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, u_D, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

# 
   # - inner(b,grad(u))*v*dx\
F = ( nu*inner(grad(u), grad(v)))*dx - f*v*dx - inner(b,grad(u))*v*dx + c*u*v*dx
a, L = lhs(F), rhs(F)


# Assemble matrix
A,load = assemble_system(a,L,bc)
#bc.apply(A)

a1PET = as_backend_type(A).mat()
ANP1 = scipy.sparse.csr_matrix(a1PET.getValuesCSR()[::-1], shape=a1PET.size)



J = 0.5*(ANP1-ANP1.T)
H = 1/(n-1) * 0.5*(ANP1+ANP1.T)


mdic = {"A": ANP1, "b": load.get_local()}
scipy.io.savemat('1d_'+str(n)+'.mat', mdic)