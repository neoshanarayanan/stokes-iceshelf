# Author: Neosha Narayanan, neosha@mit.edu
# Functionality for my Stokes flow model

import matplotlib.pyplot as plt
from dolfin import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import ufl

# ==== convert datas to numpy arrays =======
# (This is probably not useful because the data points do not have spatial tags)
def convert_to_numpy(ufl_velocity):
  ux, uy = ufl_velocity.split(deepcopy=True) # deepcopy makes a new version of the variable, instead of pointing to the original variable u
  #plot(ux, title="ux")
  ux_nodal_values = ux.vector().get_local()
  uy_nodal_values = uy.vector().get_local()

# ======= plot with colorbar ==============
def plot_with_colorbar(ufl_format_data, title):
  fig = plt.figure(figsize=(8,7))
  ax = fig.add_subplot(111)
  myplot = plot(ufl_format_data, title=title)
  divider = make_axes_locatable(plt.gca())
  cax = divider.append_axes('right', "5%", pad="3%")
  cbar = plt.colorbar(myplot, cax=cax)
  plt.show()


# ========== vector functions ============
def angle_to_components(theta, magn):
  vector = as_vector((magn*cos(theta), magn*sin(theta)))
  return(vector)

def unit_vector(ufl_vectorfield): # takes a ufl vector field (e.g., velocity field) and returns the field in unit vector form
  v1 = ufl_vectorfield[0]
  v2 = ufl_vectorfield[1]
  print("line 36, stokestools.py")
  print(v1, v2)
  magn = ufl.sqrt(v1**2 + v2**2)
  unitvectorfield = (1/magn)*ufl_vectorfield
  return(unitvectorfield)
