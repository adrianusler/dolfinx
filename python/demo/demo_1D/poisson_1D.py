import numpy as np
import matplotlib.pyplot as plt

import mpi4py
from ufl import dx, TrialFunction, TestFunction
from dolfinx.fem import FunctionSpace, Function
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_interval

def rho_func(x):# function that descibes the charge density (in arbitrary units) as a function of space
    return np.exp(-(x[0]-.5)**2)-np.exp(-(x[0]+.5)**2)

mycomm = mpi4py.MPI.COMM_WORLD
mymesh = create_interval(mycomm, 1000, [-5, 5])# mesh on interval [-5, 5] with 200 elements(?) of equal length
V = FunctionSpace(mymesh, ("CG", 1))# Function space defined on the mesh: "CG" stands for continuous Galerkin, 1 stands for the degree of the (linear) interpolation
rho = Function(V)# Function object for the charge density
rho.interpolate(rho_func)# evaluate the function rho_func on the FunctionSpace V

du = TrialFunction(V)# trial function for the variational form
v  = TestFunction(V)#  test function  for the variational form
a  = du.dx(0)*v.dx(0)*dx# left-hand side of the variational form: corresponds to -d^2u/dx^2
L  = rho*v*dx# right-hand side of the variational form: corresponds to rho
problem = LinearProblem(a, L, bcs=[])# define linear variational problem with left-hand side a and right-hand side L
u  = problem.solve()
plt.plot(u.x.array)
plt.savefig('poisson1D.png')
plt.show()
''' .....DONE..... '''
''' .............. '''
''' .....apply Dirichlet Boundary condition..... '''
from dolfinx.mesh import locate_entities_boundary
from dolfinx.fem import locate_dofs_topological, dirichletbc
leftedge_func = lambda x: np.isclose(x[0], -5)# left edge is x-value -5, leftedge_func returns True when x[0]==-5
facets_left = locate_entities_boundary(mymesh, dim=mymesh.topology.dim-1, marker=leftedge_func)
dofs_left = locate_dofs_topological(V=V, entity_dim=0, entities=facets_left)
bc_left = dirichletbc(value=0., dofs=dofs_left, V=V)
problem = LinearProblem(a, L, bcs=[bc_left])# define linear variational problem with left-hand side a and right-hand side L
u  = problem.solve()
plt.clf()
plt.plot(u.x.array)
plt.savefig('poisson1D_with_dirichletbc.png')
plt.show()