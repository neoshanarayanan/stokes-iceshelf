# Author: Neosha Narayanan, MIT Department of Materials Science & Engineering
# Contact: Neosha Narayanan, neosha@mit.edu

# FEM method from Johan Hoffman, KTH Stockholm, Sweden
# (https://github.com/johanhoffman/DD2365-VT20/blob/a6b1a551b159d887513e46cf65761d2e789c9977/template-report-Stokes.ipynb)

from dolfin import *
from mshr import *
import ufl 
import stokestools as st

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
radius = 25
# Generate mesh (examples with and without a hole in the mesh) 
resolution = 50
#mesh = RectangleMesh(Point(0.0, 0.0), Point(L, H), L*resolution, H*resolution)
#mesh = generate_mesh(Rectangle(Point(0.0,0.0), Point(L,H)) - Circle(Point(75,0.5*H),20), resolution)
mesh = generate_mesh(Rectangle(Point(0.0,0.0), Point(L,H)) - Circle(Point(60,0.25*H),radius) - Circle(Point(130, 0.75*H), radius), resolution)
#mesh = generate_mesh(Rectangle(Point(0.0,0.0), Point(L,H)) - Circle(Point(100,0.25*H),25) - Circle(Point(100, 0.75*H), 25), resolution)
#mesh = generate_mesh(Rectangle(Point(0.0,0.0), Point(L,H)) - Circle(Point(50,0.25*H),20) - Circle(Point(50, 0.75*H), 20), resolution)
#mesh = generate_mesh(Rectangle(Point(0.0,0.0), Point(L,H)) - Circle(Point(1.5,0.25*H),0.2) - Circle(Point(0.5,0.5*H),0.2) - Circle(Point(2.0,0.75*H),0.2), resolution)
print("type of mesh = " + str(type(mesh)))
#File("mesh.xml.gz") << mesh

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
# Generate mixed finite element spaces for velocity and pressure (Taylor-Hood method)
VE = VectorElement("CG", mesh.ufl_cell(), 2)
QE = FiniteElement("CG", mesh.ufl_cell(), 1)
WE = VE * QE

W = FunctionSpace(mesh, WE)
V = FunctionSpace(mesh, VE)
Q = FunctionSpace(mesh, QE)

# Define trial and test functions
w = Function(W)
(u, p) = (as_vector((w[0], w[1])), w[2])
(v, q) = TestFunctions(W)

# Examples of inflow and outflow conditions
XMIN = 0.0; XMAX = L
YMIN = 0.0; YMAX = H
uin = Expression(("4*(x[1]*(YMAX-x[1]))/(YMAX*YMAX)", "0."), YMAX=YMAX, element = V.ufl_element()) 
#uin = Expression(("-sin(x[1]*pi)", "0.0"))

# Inflow boundary (ib), outflow boundary (ob) and wall boundary (wb)
ib = Expression("near(x[0],XMIN) ? 1. : 0.", XMIN=XMIN, element = Q.ufl_element())
ob = Expression("near(x[0],XMAX) ? 1. : 0.", XMAX=XMAX, element = Q.ufl_element()) 
eps = 1e-5
wb = Expression("x[0] > XMIN + DOLFIN_EPS && x[0] < XMAX - DOLFIN_EPS ? 1. : 0.", XMIN=XMIN, XMAX=XMAX, element = Q.ufl_element())


# Defining variables for penalty formulation
h = CellDiameter(mesh)
C = 1.0e3
gamma = C/h
f = Expression(("0.0","0.0"), element = V.ufl_element())

eta = 50000
# Define variational problem on residual form: r(u,p;v,q) = 0
residual = ( - p*div(v)*dx + inner(grad(u), grad(v))*dx + div(u)*q*dx + 
            gamma*(ib*inner(u - uin, v) + wb*inner(u, v))*ds - inner(f, v)*dx )

# Solve algebraic system using iterative solver
solve(residual == 0, w)
u, p = w.split(deepcopy = True) # split u and p using deepcopy (create copies of u and p)

u = u*400/31536000 # to slow down the velocity to more realistic values
u1 = project(u, V)
p1 = project(p, Q)

# ===================== PLOT VELOCITY AND PRESSURE SOLUTIONS =======================

st.plot_with_colorbar(u1, "Velocity")
st.plot_with_colorbar(p1, "Pressure")

# ======================== CALCULATE STRESS AND STRAIN TENSORS  ===================
#Some definitions
d = u.geometric_dimension() #5* 10^12
I = Identity(2)

viscosityLabel = " "
if eta>100000:
  viscosityLabel = "(HIGH viscosity)"
elif eta<100000:
  viscosityLabel = "(LOW viscosity)"

# from Evan Cummings' thesis, page 22
def epsilon(u): # strain rate tensor
  return 0.5*(grad(u) + grad(u).T)

def sigma(u,p): # Cauchy stress tensor
  return 2*eta*epsilon(u) - p*I # from https://fenicsproject.org/pub/tutorial/html/._ftut1008.html 

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

velocityUnitVector = st.unit_vector(u1)

def stressFlowMetric(velocity_unit_vector, ps1):
  return(abs(dot(velocity_unit_vector, ps1)))

sfm = stressFlowMetric(velocityUnitVector, stress_vector_field)
st.plot_with_colorbar(sfm, "Stress Flow Metric")

