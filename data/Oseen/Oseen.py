"""This demo solves the Stokes equations, using quadratic elements for
the velocity and first degree elements for the pressure (Taylor-Hood
elements). The sub domains for the different boundary conditions
used in this simulation are computed by the demo program in
src/demo/mesh/subdomains."""

# Copyright (C) 2007 Kristian B. Oelgaard
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#~~
# Modified by Anders Logg, 2008-2009.
#
# First added:  2007-11-16
# Last changed: 2009-11-26
# Begin demo


import matplotlib.pyplot as plt
from dolfin import *
import scipy.sparse.linalg
import scipy.sparse
import scipy.io
import numpy.linalg

class MyVectorExpression(UserExpression):
    def eval(self, value, x):
        value[0] = 1
        value[1] = 1
    
    def value_shape(self):
        return (2,)

def fenics_to_numpy(A):
    Z = as_backend_type(A).mat()
    return scipy.sparse.csr_matrix(Z.getValuesCSR()[::-1], shape=Z.size)

# Initialize the vector-valued expression
b = MyVectorExpression(degree=2)

# Load mesh and subdomains
MeshN =26 #[6]; %,11,15,21,26,31];
mesh = UnitSquareMesh(MeshN,MeshN) 

# Define function spaces
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = P2 * P1
W = FunctionSpace(mesh, TH)

V = FunctionSpace(mesh,P2)
P = FunctionSpace(mesh,P1)

noslip = Constant((0, 0))

def boundary1(x, on_boundary):
    if abs(x[0]) + abs(x[1]) >= 2*DOLFIN_EPS:
        return on_boundary


bcs = DirichletBC(W.sub(0), noslip, boundary1)

dummy = Constant(0)

# Define variational problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
f = Constant((1, 0))
a = (inner(grad(u), grad(v)) - div(v)*p + q*div(u)+ inner(dot(b, grad(u)), v))*dx # + skewpart * inner(grad(u),v)
L = inner(f, v)*dx

# Compute solution
A, load = assemble_system(a, L, bcs)
a1PET = as_backend_type(A).mat()
ANP1 = scipy.sparse.csr_matrix(a1PET.getValuesCSR()[::-1], shape=a1PET.size)


bc2 = DirichletBC(V, noslip, boundary1)
pin_point = Point(0.0, 0.0)

def origin(x,on_boundary):
    if abs(x[0]) + abs(x[1]) < 2*DOLFIN_EPS:
        #print("tag zamm")
        return on_boundary

bc_p = DirichletBC(P, 0., origin, method="pointwise")

bcfull = [bc2,bc_p]


# other way; top left block
u = TrialFunction(V)
v = TestFunction(V)
a_top_left = (inner(grad(u), grad(v)) + inner(dot(b, nabla_grad(u)), v))*dx 
L = inner(f,v)*dX

print("A")

A, load = assemble_system(a_top_left, L, bcfull)
A2 = fenics_to_numpy(A)

print("symm part ", numpy.linalg.norm((A2+A2.T).toarray()))
print("skew part ", numpy.linalg.norm((A2-A2.T).toarray()))


p = TestFunction(P)
v = TrialFunction(V)
a_bottom_left = div(v)*p*dx # + skewpart * inner(grad(u),v)
L = dummy*p*dx

print("B")

B, loadb = assemble_system(a_bottom_left, L, bcfull)
B = fenics_to_numpy(B)

v = TestFunction(V)
q = TrialFunction(P)
a_top_right = -inner(grad(q),v)*dx #inner(grad(q),v)*dx# # + skewpart * inner(grad(u),v)
L2 = dummy*div(v)*dx

print("Bt")

Bt, loadb2 = assemble_system(a_top_right, L2, bcfull)
Bt = fenics_to_numpy(Bt)


Z = scipy.sparse.bmat( [[A2,-B.T], [B, None] ] )
Z2 = scipy.sparse.bmat( [[A2,Bt], [B, None] ] )

J = 0.5*(Z - Z.T)
R = 0.5*(Z + Z.T)

J2 = 0.5*(Z2-Z2.T)
R2 = 0.5*(Z2+Z2.T)

tol = 0.5

p = TrialFunction(P)
q = TestFunction(P)

mass_pressure = p*q*dx
ell = dummy*q*dx

Mp,dummy = assemble_system(mass_pressure,ell,bcfull)
Mp = fenics_to_numpy(Mp)



# #observation_region = Expression('x[0] >=  tol || x[1] >= tol ? 1 : 0', degree=0,tol=tol)
# control_region = Expression('x[0] <=  tol || x[1] <= tol ? 1 : 0', degree=0,tol=tol)     
 
# mass = inner(u,v)*dx
# b = inner(u,v)*control_region*dx
# c = inner(u,v)*dx 
 
# B,dum = assemble_system(b,L,bcs)
# C = B
# Mass = B
# #C,dum = assemble(c,L,bcs)
# #Mass,dum = assemble(mass,L,bcs)

 
# BPET = as_backend_type(B).mat()
# BNP = scipy.sparse.csr_matrix(BPET.getValuesCSR()[::-1], shape=BPET.size)

# CPET = as_backend_type(C).mat()
# CNP = scipy.sparse.csr_matrix(CPET.getValuesCSR()[::-1], shape=CPET.size)

# MASSPET = as_backend_type(Mass).mat()
# MassNP = scipy.sparse.csr_matrix(MASSPET.getValuesCSR()[::-1], shape=MASSPET.size)

mdic = {"A": Z, "b": load.get_local().T, "A11": A2, "B": B, "Mp": Mp}

scipy.io.savemat('Oseen_'+str(MeshN)+'.mat', mdic)
