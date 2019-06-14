'''
Representing spatial structure and calculating distances
'''
from pyNN.space import *
#The simplest structure is a grid, whether 1D, 2D or 3D
line = Line(dx=100.0, x0=0.0, y=200.0, z=500.0)
print(line.generate_positions(7))
grid = Grid2D(aspect_ratio=3, dx=10.0, dy=25.0, z=-3.0)
#an x:y ratio =3 - shape of gird
print(grid.generate_positions(3))
print(grid.generate_positions(12))
#default- iterating first over the z dimension, then y, then x
rgrid = Grid2D(aspect_ratio=1, dx=10.0, dy=10.0, fill_order='random', rng=NumpyRNG(seed=13886))
print(rgrid.generate_positions(9))
#fill the grid randomly

glomerulus = RandomStructure(boundary=Sphere(radius=200.0), rng=NumpyRNG(seed=34534))
#RandomStructure - distributes neurons randomly and uniformly within a given volume
#volume classes - Sphere, Cuboid
print(glomerulus.generate_positions(5))

#=====Defining own Structure classes=====
#inherit from BaseStructure and implement a generate_positions() method
class MyStructure(BaseStructure):
    parameter_names = ("spam", "eggs")

    def __init__(self, spam=3, eggs=1):
        pass

    def generate_positions(self, n):
        pass
        # must return a 3xn numpy array
#To define your own Shape class for use with RandomStructure
#subclass Shape and implement a sample() method
class Tetrahedron(Shape):

    def __init__(self, side_length):
        pass

    def sample(self, n, rng):
       pass
       # return a nx3 numpy array.

