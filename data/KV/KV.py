from fenics import *
from dolfin import *

import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import matplotlib.pyplot as plt

# --- Safe PETSc -> SciPy CSR conversion
def fenics_to_csr(A):
    Ap = as_backend_type(A).mat()
    indptr, indices, data = Ap.getValuesCSR()
    return sp.csr_matrix((data, indices, indptr), shape=Ap.getSize())

MeshN = 51
mesh = UnitIntervalMesh(MeshN)

# Choose ONE of these boundary descriptions:

# (A) left boundary only (x=0)
def boundary_left(x, on_boundary):
    return on_boundary and near(x[0], 0.0)

# (B) all boundary (both ends)
def boundary_all(x, on_boundary):
    return on_boundary

# Pick which one you want:
boundary_fn = boundary_left

P = FunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 2)

f = Constant(1.0)
kvdamping = Constant(1.0)

u_D = Constant(0.0)
bcp = DirichletBC(P, u_D, boundary_fn)
bcq = DirichletBC(Q, u_D, boundary_fn)  # only needed if you ever assemble forms on Q with Dirichlet constraints
bcf = [bcp,bcq]

# ---------- A11 block (symmetric SPD, uses FIRST derivatives)
p = TrialFunction(P)
v = TestFunction(P)

# If you intended a 'stiffness-like' block on P:
a_top_left = kvdamping * p.dx(0) * v.dx(0) * dx
L_dummy = Constant(0.0) * v * dx  # zero RHS just to use assemble_system for BC application

A11, _ = assemble_system(a_top_left, L_dummy, bcf)   # apply BCs on P
#A11 = assemble(a_top_left)
A11_csr = fenics_to_csr(A11)

# ---------- B block (coupling P->Q). You had (p' * w'), keep that but apply BCs on TRIAL space P.
p = TrialFunction(P)
w = TestFunction(P)

a_bottom_left = p.dx(0) * w.dx(0) * dx
L_dummy = Constant(0.0) * w * dx

Bmat, _ = assemble_system(a_bottom_left,L_dummy,bcf)  # constrain P only
B_csr = fenics_to_csr(Bmat)

# ---------- Mass on P
p = TrialFunction(P)
q = TestFunction(P)
Mp_form = p * q * dx
Mp = assemble(Mp_form)
# Apply Dirichlet only if you want the constrained-mass (often you do when comparing spectra on constrained subspace):
b_dummy = assemble(Constant(0.0) * q * dx)
bcp.apply(Mp, b_dummy)
Mp_csr = fenics_to_csr(Mp)

# ---------- Diagnostics
print("A11 is symmetric?  ||A - A^T|| =", np.linalg.norm((A11_csr - A11_csr.T).toarray()))
print("B shape:", B_csr.shape, "  Mp shape:", Mp_csr.shape)


# Example eigenvalue checks (dense for small N only!)
print("eig(A11) (small N):", np.linalg.eigvals(A11_csr.toarray()))
print("eig(B) (small N):", np.linalg.eigvals(B_csr.toarray()))

print("singular values of B (small N):", np.linalg.svd(B_csr.toarray(), compute_uv=False))

#A11_csr = A11_csr*np.linalg.inv(Mp_csr.toarray())*A11_csr
# ---------- Save to MAT
sio.savemat(f"KV_{MeshN}.mat", {
    "A11": A11_csr,
    "B": B_csr,
    "Mp": Mp_csr
})
