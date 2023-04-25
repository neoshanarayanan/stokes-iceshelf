from mshr import *
from dolfin import *
import matplotlib.pyplot as plt

L = 400
H = 400
XMIN = 0
XMAX = L
YMIN = 0
YMAX = H
radius = 25
center1x, center1y = 60, 0.25*H
center2x, center2y = 130, 0.75*H

# Generate mesh (examples with and without a hole in the mesh) 
resolution = 50
#mymesh = RectangleMesh(Point(0.0, 0.0), Point(L, H), L*resolution, H*resolution)
#mymesh = generate_mesh(Rectangle(Point(0.0,0.0), Point(L,H)) - Circle(Point(150,0.5*H),80), resolution)
mymesh = generate_mesh(Rectangle(Point(0.0,0.0), Point(L,H)) - Circle(Point(60,0.25*H),radius) - Circle(Point(130, 0.75*H), radius), resolution)
#mesh = generate_mesh(Rectangle(Point(0.0,0.0), Point(L,H)) - Circle(Point(50,0.25*H),20) - Circle(Point(50, 0.75*H), 20), resolution)
#mesh = generate_mesh(Rectangle(Point(0.0,0.0), Point(L,H)) - Circle(Point(1.5,0.25*H),0.2) - Circle(Point(0.5,0.5*H),0.2) - Circle(Point(2.0,0.75*H),0.2), resolution)

File("meshes/mesh.xml.gz") << mymesh
mesh = Mesh("meshes/mesh.xml.gz")

plt.figure()
plot(mesh)
plt.show()


dolfin_eps = 1E-10


def distance(x0, x1, centerx, centery):
    distance = sqrt((centerx-x0)**2 + (centery-x1)**2)
    return(distance)


# Sub domain for no-slip (mark whole boundary, inflow and outflow will overwrite)
class Noslip(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and distance(x[0], x[1], center1x, center1y)<radius + dolfin_eps) or (on_boundary and distance(x[0], x[1], center2x, center2y)<radius + dolfin_eps)


'''
# Sub domain for no-slip (mark whole boundary, inflow and outflow will overwrite)
class Noslip(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary 
'''

# Sub domain for inflow (right)
class Inflow(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > 1.0 - DOLFIN_EPS and on_boundary

# Sub domain for outflow (left)
class Outflow(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < DOLFIN_EPS and on_boundary

# Create mesh functions over the cell facets
sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)

# Mark all facets as sub domain 3
sub_domains.set_all(3)

# Mark no-slip facets as sub domain 0, 0.0
noslip = Noslip()
noslip.mark(sub_domains, 0)

# Mark inflow as sub domain 1, 01
inflow = Inflow()
inflow.mark(sub_domains, 1)

# Mark outflow as sub domain 2, 0.2, True
outflow = Outflow()
outflow.mark(sub_domains, 2)

# Save sub domains to file
file = File("meshes/subdomains.xml.gz")
file << sub_domains

