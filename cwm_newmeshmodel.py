# Author: Neosha Narayanan, MIT Department of Materials Science & Engineering
# Contact: Neosha Narayanan, neosha@mit.edu

# FEM method from Johan Hoffman, KTH Stockholm, Sweden
# (https://github.com/johanhoffman/DD2365-VT20/blob/a6b1a551b159d887513e46cf65761d2e789c9977/template-report-Stokes.ipynb)

from dolfin import *
from mshr import *
import ufl 
import stokestools as st
import meshmaker

import dolfin.common.plotting as fenicsplot
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from decimal import *
import math
import numpy as np


# =========================== DEFINE MESH ==========================
# Dimensions of rectangular domain (m)
L = 400
H = 400

# Generate mesh (examples with and without a hole in the mesh) 
resolution = 50
#mesh = RectangleMesh(Point(0.0, 0.0), Point(L, H), L*resolution, H*resolution)
#mesh = generate_mesh(Rectangle(Point(0.0,0.0), Point(L,H)) - Circle(Point(75,0.5*H),20), resolution)
#mesh = generate_mesh(Rectangle(Point(0.0,0.0), Point(L,H)) - Circle(Point(60,0.25*H),25) - Circle(Point(130, 0.75*H), 25), resolution)
mesh = generate_mesh(Rectangle(Point(0.0,0.0), Point(L,H)) - Circle(Point(75,0.5*H),20), resolution)

#mesh = generate_mesh(Rectangle(Point(0.0,0.0), Point(L,H)) - Circle(Point(50,0.25*H),20) - Circle(Point(50, 0.75*H), 20), resolution)
#mesh = generate_mesh(Rectangle(Point(0.0,0.0), Point(L,H)) - Circle(Point(1.5,0.25*H),0.2) - Circle(Point(0.5,0.5*H),0.2) - Circle(Point(2.0,0.75*H),0.2), resolution)

'''
mesh = meshmaker.generateMesh()
meshmaker.createAndSaveMesh(mesh)
sub_domains = MeshFunction("size_t", mesh, "~/meshes/subdomains.xml.gz")
'''

# Local mesh refinement (specified by a cell marker)
no_levels = 0
for i in range(0,no_levels):
  cell_marker = MeshFunction("bool", mesh, mesh.topology().dim())
  for cell in cells(mesh):
    cell_marker[cell] = False
    p = cell.midpoint()
    if p.distance(Point(1.5, 0.5)) < 1.0:
        cell_marker[cell] = True
  mesh = refine(mesh, cell_marker)

# Plot mesh
plt.figure()
plot(mesh, title="Mesh visualization")
plt.show()


# =====================DEFINE BOUNDARY CONDITIONS, SOLVE PDEs =======================



inflow   = 'near(x[0], 0)'
outflow  = 'near(x[0], 400)'
walls    = 'near(x[1], 0) || near(x[1], 400)'
cylinder = 'on_boundary && x[0]>50 && x[0]<100 && x[1]>175 && x[1]<225'

noslip = Constant((0, 0))
# Generate mixed finite element spaces for velocity and pressure (Taylor-Hood method)
VE = VectorElement("CG", mesh.ufl_cell(), 2)
QE = FiniteElement("CG", mesh.ufl_cell(), 1)
WE = VE * QE


# inflow boundary at x = 1
XMIN = 0.0; XMAX = L
YMIN = 0.0
YMAX = H

V = FunctionSpace(mesh, VE)
W = FunctionSpace(mesh, WE)
Q = FunctionSpace(mesh, QE)
inflow_expression = Expression(("4*(x[1]*(YMAX-x[1]))/(YMAX*YMAX)", "0."), YMAX=YMAX, element = V.ufl_element())
#bc0 = DirichletBC(W.sub(0), noslip, sub_domains, 0)
bcu_inflow = DirichletBC(W.sub(0), inflow_expression, inflow)
bcu_walls = DirichletBC(W.sub(0), noslip, walls)
bcu_cylinder = DirichletBC(W.sub(0), noslip, cylinder)
bcp_outflow = DirichletBC(Q, Constant(0), outflow)
bcs = [bcu_inflow, bcu_walls, bcu_cylinder, bcp_outflow]



#bc1 = DirichletBC(W.sub(0), inflow, sub_domains, 1)
#bcs = [bc0, bc1]
eta = 1
I = Identity(2)

# from Evan Cummings' thesis, page 22
def epsilon(u): # strain rate tensor
  return 0.5*(grad(u) + grad(u).T)

def sigma(u,p): # Cauchy stress tensor
  return 2*eta*epsilon(u) - p*I # from https://fenicsproject.org/pub/tutorial/html/._ftut1008.html 


# Define trial and test functions
w = Function(W)
#(u, p) = (as_vector((w[0], w[1])), w[2])
#(v, q) = TestFunctions(W)
u, p = TrialFunctions(W)
v, q = TestFunctions(W)
f = Constant((0, 0)) # set f, body force, to 0
a = inner(sigma(u,p), grad(v))*dx + q*div(u)*dx
L = inner(f, v)*dx

A = assemble(a)
print(A.size(0), A.size(1))
b = assemble(L)
for bc in bcs:
    bc.apply(a, L)
    solve(A, w.vector(), b)


#solve(a==L, w, bcs)

u, p = w.split(True)


# ===================== PLOT VELOCITY AND PRESSURE SOLUTIONS =======================

st.plot_with_colorbar(u, "Velocity")
st.plot_with_colorbar(p, "Pressure")

# ======================== CALCULATE STRESS AND STRAIN TENSORS  ===================
#Some definitions
d = u.geometric_dimension()
eta = 5000000000000 #5* 10^12
I = Identity(2)

viscosityLabel = " "
if eta>100000:
  viscosityLabel = "(HIGH viscosity)"
elif eta<100000:
  viscosityLabel = "(LOW viscosity)"

# ========================= STRESS INTENSITY CALCULATION =========================
s = sigma(u, p) - (1/3)*tr(sigma(u, p))*Identity(d) # deviatoric stress
von_Mises = sqrt(3./2*inner(s, s)) # von Mises stress = sqrt(3/2(s:s))

VM = FunctionSpace(mesh, 'P', 1)
von_Mises = project(von_Mises, VM)


#vm_nodal_values = von_Mises.vector().get_local()
#print("vm_nodal_values = " + str(vm_nodal_values))
#coordinates = mesh.coordinates()


st.plot_with_colorbar(von_Mises, "von Mises stress intensity")


# ========================= ANALYZE PRINCIPAL STRAINS ============================
TS = TensorFunctionSpace(mesh, "CG", 2)
strain_rate_tensor = project(epsilon(u), TS)
ex = strain_rate_tensor[0, 0]
ey = strain_rate_tensor[1, 1]
gxy = strain_rate_tensor[0, 1]


def get_e2_vectorfield(ex, ey, gxy):
  angle1 = atan(((2*gxy)/ex-ey)/2)
  angle2 = angle1 + pi/2

  # equation (10-5) in Mechanics of Materials textbook
  def formula(ex, ey, exy, theta):
    ex_prime = (ex+ey)/2 + ((ex-ey)/2)*cos(2*theta) + (exy/2)*sin(2*theta)
    return(ex_prime)

  magn1 = formula(ex, ey, gxy, angle1)
  magn2 = formula(ex, ey, gxy, angle2)

  e2 = ufl.conditional(ufl.le(magn1, magn2), st.angle_to_components(angle1, magn1), st.angle_to_components(angle2, magn2))

  return(e2)

e2 = get_e2_vectorfield(ex, ey, gxy)
st.plot_with_colorbar(e2, "Second Principal Strain Rate")


# Just magnitudes of principal strains purely

def e2_magnitude(ex, ey, gxy): # get the magnitude of the second principal strain rate
  emin = (ex+ey)/2 - sqrt(((ex-ey)/2)**2 + (gxy)**2)
  return(emin)

def e1_magnitude(ex, ey, gxy):
  emax = (ex+ey)/2 + sqrt(((ex-ey)/2)**2 + (gxy)**2)
  return(emax)


e2_magn = e2_magnitude(ex, ey, gxy)
st.plot_with_colorbar(e2_magn, "Magnitude of Minimum Strain Rate")


# ========================== ANALYZE PRINCIPAL STRESSES ============================

# Extract values from the stress tensor
TS = TensorFunctionSpace(mesh, "CG", 2) # not sure why we chose 2 here
cauchy = project(sigma(u,p), TS)
sx = cauchy[0,0]
sy = cauchy[1, 1]
txy = cauchy[0, 1]

# Direction of maximum tensile stress, as calculated in Introduction to Contact Mechanics textbook
# Returns the X and Y components of the maximum tensile stress
def s1_components(sx, sy, txy, a):
  theta = ((2*txy)/(sx-sy))/2
  return st.angle_to_components(theta, a)

# Magnitude of maximum tensile stress, as calculated in Introduction to Contact Mechanics textbook
def sigma1_magnitude(sx, sy, txy): # magnitude of first principal stress
  return((sx+sy)/2 + sqrt(((sx-sy)/2)**2 + txy**2))

'''
def get_s2_vectorfield(sx, sy, txy):
  angle1 = atan(((2*txy)/sx-sy)/2)
  angle2 = angle1 + pi/2

  # equation (10-5) in Mechanics of Materials textbook
  def formula(sx, sy, txy, theta):
    sx_prime = (sx+sy)/2 + ((sx-sy)/2)*cos(2*theta) + (txy/2)*sin(2*theta) # gives the magnitude based on the angle
    return(sx_prime)

  magn1 = formula(sx, sy, txy, angle1)
  magn2 = formula(sx, sy, txy, angle2)

  e2 = ufl.conditional(ufl.le(magn1, magn2), st.angle_to_components(angle1, magn1), st.angle_to_components(angle2, magn2))

  return(e2)
'''

# Generate vector field showing direction and magnitude of principal stresses
def sigma1(sx, sy, txy): 
  a = sigma1_magnitude(sx, sy, txy)
  return(s1_components(sx, sy, txy, a))
stress_vector_field = sigma1(sx, sy, txy) #get_s2_vectorfield(sx, sy, txy)

plot(stress_vector_field, title="Maximum Tensile Stress ")#+viscosityLabel)
plt.show()

# ================================ Stress flow angle ================================

def stressFlowAngle(th, u): 
  vectorValuedAngle = u-th
  scalarValuedAngle = abs(atan(vectorValuedAngle[1]/vectorValuedAngle[0]))
  scalarValuedAngleDegrees = 180*scalarValuedAngle/3.14159265
  return(scalarValuedAngleDegrees)

ps1_angle = s1_components(sx, sy, txy, sigma1_magnitude(sx, sy, txy))
sfa = stressFlowAngle(ps1_angle, u)

fig3 = plt.figure(figsize=(8,7))
ax3 = fig3.add_subplot(111)
stressflowangle = plot(sfa, title = "Stress Flow Angle (degrees) "+viscosityLabel)
divider3 = make_axes_locatable(plt.gca())
cax = divider3.append_axes('right', "5%", pad="3%")
cbar = plt.colorbar(stressflowangle, cax=cax, cmap = 'gist_earth')
plt.show()

# =========================== revised stress flow angle ['stress flow metric']============================
# dot unit vector of velocity with the maximum tensile stress, then investigate!! 

velocityUnitVector = st.unit_vector(u)

def stressFlowMetric(velocity_unit_vector, ps1):
  return(dot(velocity_unit_vector, ps1))

sfm = stressFlowMetric(velocityUnitVector, stress_vector_field)
st.plot_with_colorbar(sfm, "Stress Flow Metric")

