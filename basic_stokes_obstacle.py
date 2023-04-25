from dolfin import *
import matplotlib.pyplot as plt
from mshr import *
import dolfin.common.plotting as fenicsplot
import ufl
import stokestools as st


L = 400
H = 400
YMAX = H

mesh = Mesh("meshes/mesh.xml.gz")
sub_domains = MeshFunction("size_t", mesh, "meshes/subdomains.xml.gz")

# Define Taylor-Hood function space
VE = VectorElement("CG", mesh.ufl_cell(), 2)
QE = FiniteElement("CG", mesh.ufl_cell(), 1)
WE = VE * QE

W = FunctionSpace(mesh, WE)
V = FunctionSpace(mesh, VE)
Q = FunctionSpace(mesh, QE)

noslip = Constant((0, 0))
bc0 = DirichletBC(W.sub(0), noslip, sub_domains, 0) # subdomain marked 0 (noslip)

# Inflow boundary condition for velocity
# x0 = 1
inflow = Expression(("4*(x[1]*(YMAX-x[1]))/(YMAX*YMAX)", "0."), YMAX=YMAX, element = V.ufl_element())
bc1 = DirichletBC(W.sub(0), inflow, sub_domains, 1) # subdomain marked 1 (inflow)

# Boundary condition for pressure at outflow
# x0 = 0
zero = Constant(0)
bc2 = DirichletBC(W.sub(1), zero, sub_domains, 2) # subdomain marked 2 (outflow)

# Collect boundary conditions
bcs = [bc0, bc1, bc2]
eta = 1
I = Identity(2)
def epsilon(u): # strain rate tensor
  return 0.5*(grad(u) + grad(u).T)

def sigma(u,p): # Cauchy stress tensor
  return 2*eta*epsilon(u) - p*I # from https://fenicsproject.org/pub/tutorial/html/._ftut1008.html 

# Define trial and test functions
u, p = TrialFunctions(W)
v, q = TestFunctions(W)
f = Constant((0, 0)) # set f, body force, to 0
a = inner(sigma(u,p), grad(v))*dx + q*div(u)*dx
L = inner(f, v)*dx

# Compute solution
w = Function(W)
solve(a == L, w, bcs)

# Split the mixed solution using deepcopy
# (needed for further computation on coefficient vector)
(u, p) = w.split(True)

u = project(u,V)
st.plot_with_colorbar(u, "Velocity")
st.plot_with_colorbar(p, "Pressure")


