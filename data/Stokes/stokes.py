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
#
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

# Load mesh and subdomains
MeshN = 51
mesh = UnitSquareMesh(MeshN,MeshN) #Mesh("../dolfin_fine.xml.gz")
#sub_domains = MeshFunction("size_t", mesh, "../dolfin_fine_subdomains.xml.gz")


#plt.figure()
#plot(mesh)

#plt.figure()
#plot(sub_domains)

# Define function spaces
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = P2 * P1
W = FunctionSpace(mesh, TH)

# No-slip boundary condition for velocity
# x1 = 0, x1 = 1 and around the dolphin
noslip = Constant((0, 0))
stab = Constant(1)
stab1 = Constant(1) # necessary soonst kein guter precond
stab2 = Constant(1) # necessary sonst nicht pos def
#bc0 = DirichletBC(W.sub(0), noslip, sub_domains, 0)

# Inflow boundary condition for velocity
# x0 = 1
inflow = Expression(("-sin(x[1]*pi)", "0.0"), degree=2)
#bc1 = DirichletBC(W.sub(0), inflow, sub_domains, 1)


def boundary1(x, on_boundary):
     #return False #on_boundary
     #if x[1] == 0:
     #    return on_boundary
     #if x[1] == 1:
     return on_boundary
     # return false
bcs = DirichletBC(W.sub(0), noslip, boundary1)


# def boundary0(x, on_boundary):
#      #return False #on_boundary
#      if x[0] == 1:
#          return on_boundary
#      # return false
# bc0 = DirichletBC(W.sub(0), inflow, boundary0)


# Collect boundary conditions
#bcs = [bc0, bc1]

# Define variational problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
f = Constant((0, 0))
#a = (inner(grad(u), grad(v)) - div(v)*p + q*div(u)+ p*q)*dx
a = (stab*inner(u,v) + inner(grad(u), grad(v)) - div(v)*p + q*div(u)+ \
     (stab1*inner(grad(p),grad(q)) + stab2*p*q))*dx

L = inner(f, v)*dx

# Compute solution
A, load = assemble_system(a, L, bcs)


a1PET = as_backend_type(A).mat()
ANP1 = scipy.sparse.csr_matrix(a1PET.getValuesCSR()[::-1], shape=a1PET.size)

tol = 0.5

#observation_region = Expression('x[0] >=  tol || x[1] >= tol ? 1 : 0', degree=0,tol=tol)
#control_region = Expression('x[0] <=  tol || x[1] <= tol ? 1 : 0', degree=0,tol=tol)     
 
mass = inner(u,v)*dx
b = inner(u,v)*dx
c = inner(u,v)*dx 
 
B,dum = assemble_system(b,L,bcs)
C = B
Mass = B
#C,dum = assemble(c,L,bcs)
#Mass,dum = assemble(mass,L,bcs)

 
BPET = as_backend_type(B).mat()
BNP = scipy.sparse.csr_matrix(BPET.getValuesCSR()[::-1], shape=BPET.size)

CPET = as_backend_type(C).mat()
CNP = scipy.sparse.csr_matrix(CPET.getValuesCSR()[::-1], shape=CPET.size)

MASSPET = as_backend_type(Mass).mat()
MassNP = scipy.sparse.csr_matrix(MASSPET.getValuesCSR()[::-1], shape=MASSPET.size)

mdic = {"A": ANP1, "b": load.get_local().T, "C": CNP, "B": BNP, "Mass": MassNP}

scipy.io.savemat('cond/stokes_anders_'+str(MeshN)+'.mat', mdic)
