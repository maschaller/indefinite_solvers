import matplotlib.pyplot as plt
from dolfin import *
import scipy.sparse.linalg
import scipy.sparse
import scipy.io
import numpy as np


def fenics_to_numpy(A):
    Z = as_backend_type(A).mat()
    return scipy.sparse.csr_matrix(Z.getValuesCSR()[::-1], shape=Z.size)


# Load mesh and subdomains
MeshN = 101
mesh = UnitSquareMesh(MeshN,MeshN) #Mesh("../dolfin_fine.xml.gz")

dummy = Constant(0)

# Define function spaces
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = P2 * P1
W = FunctionSpace(mesh, TH)

V = FunctionSpace(mesh,P2)
P = FunctionSpace(mesh,P1)

# No-slip boundary condition for velocity
# x1 = 0, x1 = 1 and around the dolphin
noslip = Constant((0, 0))
stab = Constant(0)
stab1 = Constant(1) # necessary soonst kein guter precond
stab2 = Constant(1) # necessary sonst nicht pos def
#bc0 = DirichletBC(W.sub(0), noslip, sub_domains, 0)

# Inflow boundary condition for velocity
# x0 = 1
#inflow = Expression("-sin(x[1]*pi) + x[0]^2", degree=2)
#bc1 = DirichletBC(W.sub(0), inflow, sub_domains, 1)


# full assembly
def boundary1(x, on_boundary):
     return on_boundary

bcs = DirichletBC(W.sub(0), noslip, boundary1)

# Define variational problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
f = Constant((1, 1))
#a = (inner(grad(u), grad(v)) - div(v)*p + q*div(u)+ p*q)*dx
a = (stab*inner(u,v) + inner(grad(u), grad(v)) - div(v)*p + q*div(u)+ \
     (stab1*inner(grad(p),grad(q)) + stab2*p*q))*dx

L = inner(f, v)*dx

# Compute solution
A, load = assemble_system(a, L, bcs)


a1PET = as_backend_type(A).mat()
ANP1 = scipy.sparse.csr_matrix(a1PET.getValuesCSR()[::-1], shape=a1PET.size)

### blockwise assembly

bcfull = DirichletBC(V, noslip, boundary1)

# other way; top left block
u = TrialFunction(V)
v = TestFunction(V)
a_top_left = (stab*inner(u,v) + inner(grad(u), grad(v)))*dx
L = inner(f,v)*dx

A, load = assemble_system(a_top_left, L, bcfull)
A11 = fenics_to_numpy(A)

print(np.linalg.norm(load.get_local()))

print("top left assembled")


p = TrialFunction(P)
q = TestFunction(P)
a_bottom_right = (stab1*inner(grad(p),grad(q)) + stab2*p*q)*dx
L = dummy*q*dx

A, loadd = assemble_system(a_bottom_right, L, bcfull)
A22 = fenics_to_numpy(A)

print("top left assembled")



p = TestFunction(P)
v = TrialFunction(V)
a_bottom_left = div(v)*p*dx # + skewpart * inner(grad(u),v)
L = dummy*p*dx


B, loadb = assemble_system(a_bottom_left, L, bcfull)
B = fenics_to_numpy(B)

print("bottom left assembled")


v = TestFunction(V)
q = TrialFunction(P)
a_top_right = -inner(grad(q),v)*dx #inner(grad(q),v)*dx# # + skewpart * inner(grad(u),v)
L2 = dummy*div(v)*dx


Bt, loadb2 = assemble_system(a_top_right, L2, bcfull)
Bt = fenics_to_numpy(Bt)

print("top right assembled")


Z = scipy.sparse.bmat( [[A11,-B.T], [B, A22] ] )#.toarray()
#Z2 = scipy.sparse.bmat( [[A11,-Bt], [B, A22] ] ).toarray()


Sym = scipy.sparse.bmat( [[A11,None], [None, A22] ] ).toarray()
Skew = scipy.sparse.bmat( [[None,Bt], [B, None] ] ).toarray()


#observation_region = Expression('x[0] >=  tol || x[1] >= tol ? 1 : 0', degree=0,tol=tol)
#control_region = Expression('x[0] <=  tol || x[1] <= tol ? 1 : 0', degree=0,tol=tol)     
 
u = TrialFunction(V)
v = TestFunction(V)

b = inner(u,v)*dx
L =inner(f,v)*dx

Bin,dum = assemble_system(b,L,bcs)
BPET = fenics_to_numpy(Bin)
Bfinal = scipy.sparse.bmat( [[BPET,-0*B.T], [0*B, 0*A22] ] )

p = TrialFunction(P)
q = TestFunction(P)

massp = p*q*dx
L = dummy*q*dx
massp,dum = assemble_system(massp,L,bcs)
MassPPET = fenics_to_numpy(massp)

Massfinal = scipy.sparse.bmat( [[BPET,-0*B.T], [0*B, MassPPET] ] )

###load 

loadvec = np.concatenate([load.get_local(),0*loadb])

print(np.linalg.norm(loadvec))

mdic = {"A": Z, "b": loadvec, "C": Bfinal, "B": Bfinal, "Mass": Massfinal}

scipy.io.savemat('stokes_blockwise_011_'+str(MeshN)+'.mat', mdic)
