
#%%
# Load neccessary modules.
try:
    from google.colab import files #Unnecessary if ran locally
    ! sudo apt-get install texlive-latex-extra 
    ! sudo apt install texlive-fonts-recommended
    ! sudo apt install dvipng
    ! sudo apt-get install cm-super
except:
    pass

import numpy as np
import time

# Install FEniCS
try:
    import dolfin
except ImportError as e:
    !apt-get install -y -qq software-properties-common
    !add-apt-repository -y ppa:fenics-packages/fenics
    !apt-get update -qq
    !apt install -y --no-install-recommends fenics
    !sed -i "s|#if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR <= 8 && PETSC_VERSION_RELEASE == 1|#if 1|" /usr/include/dolfin/la/PETScLUSolver.h
    !rm -rf /usr/lib/python3/dist-packages/mpi4py*
    !rm -rf /usr/lib/python3/dist-packages/petsc4py*
    !rm -rf /usr/lib/python3/dist-packages/slepc4py*
    !rm -rf /usr/lib/petsc/lib/python3/dist-packages/dolfin*
    !rm -rf /usr/lib/petsc/lib/python3/dist-packages/mshr*
    !wget "https://drive.google.com/uc?export=download&id=1cT_QBJCOW_eL3BThnval3bcpb8o0w-Ad" -O /tmp/mpi4py-2.0.0-cp37-cp37m-linux_x86_64.whl
    !wget "https://drive.google.com/uc?export=download&id=119i49bxlGn1mrnhTNmOvM4BqmjrT9Ppr" -O /tmp/petsc4py-3.7.0-cp37-cp37m-linux_x86_64.whl
    !wget "https://drive.google.com/uc?export=download&id=1-1tVfu8qz3bRC2zvR8n3RESpesWqNnn6" -O /tmp/slepc4py-3.7.0-cp37-cp37m-linux_x86_64.whl
    !wget "https://drive.google.com/uc?export=download&id=1-3qY4VIJQaXVO1HfGQIzTIURIeJbvX-9" -O /tmp/fenics_dolfin-2019.2.0.dev0-cp37-cp37m-linux_x86_64.whl
    !wget "https://drive.google.com/uc?export=download&id=1-5SMjgjMuee_9WLeYtGe8N_lvipWEN7W" -O /tmp/mshr-2019.2.0.dev0-cp37-cp37m-linux_x86_64.whl
    !pip3 install /tmp/mpi4py-2.0.0-cp37-cp37m-linux_x86_64.whl --upgrade
    !pip3 install /tmp/petsc4py-3.7.0-cp37-cp37m-linux_x86_64.whl --upgrade
    !pip3 install /tmp/slepc4py-3.7.0-cp37-cp37m-linux_x86_64.whl --upgrade
    !pip3 install /tmp/fenics_dolfin-2019.2.0.dev0-cp37-cp37m-linux_x86_64.whl --upgrade
    !pip3 install /tmp/mshr-2019.2.0.dev0-cp37-cp37m-linux_x86_64.whl --upgrade
    !pip3 -q install --upgrade sympy
    import dolfin

from dolfin import *; from mshr import *

import dolfin.common.plotting as fenicsplot

from matplotlib import pyplot as plt

#%%
# Define rectangular domain 
L = 4
H = 2

# Define circle
xc = 1.0
yc = 0.5*H
rc = 0.2

# Define subdomains (for boundary conditions)
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) 

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L)

class Lower(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0)

class Upper(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], H)
      
left = Left()
right = Right()
lower = Lower()
upper = Upper()

# Generate mesh (examples with and without a hole in the mesh) 
resolution = 32
#mesh = RectangleMesh(Point(0.0, 0.0), Point(L, H), L*resolution, H*resolution)
mesh = generate_mesh(Rectangle(Point(0.0,0.0), Point(L,H)) - Circle(Point(xc,yc),rc), resolution)

# Local mesh refinement (specified by a cell marker)
no_levels = 0
for i in range(0,no_levels):
  cell_marker = MeshFunction("bool", mesh, mesh.topology().dim())
  for cell in cells(mesh):
    cell_marker[cell] = False
    p = cell.midpoint()
    if p.distance(Point(xc, yc)) < 1.0:
        cell_marker[cell] = True
  mesh = refine(mesh, cell_marker)

# Define mesh functions (for boundary conditions)
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
left.mark(boundaries, 1)
right.mark(boundaries, 2)
lower.mark(boundaries, 3)
upper.mark(boundaries, 4)

plt.figure()
plot(mesh)
plt.show()
# %%

E = ("amp_x*sin(2.0*pi*t*freq)*sin(pi*x[0]/L) - amp_x*sin(2.0*pi*(t-dt)*freq)*sin(pi*x[0]/L)",
     "amp_y*sin(2.0*pi*t*freq-0.5*pi)*sin(pi*x[1]/H) - amp_y*sin(2.0*pi*(t-dt)*freq-0.5*pi)*sin(pi*x[1]/H)")
amp_x = 0.0; amp_y = 0.3; freq = 0.3
kwargs = {"L" : L, "H" : H, "amp_x" :amp_x, "amp_y":amp_y, "freq":freq}
deform = [E, kwargs]
mesh = generate_mesh(Rectangle(Point(0.0,0.0), Point(L,H)) - Circle(Point(xc,yc),rc), resolution)


Tmax=0.05
meshdeform=deform
uinit=None
pinit=None
tinit=0.0
psi=None
# Generate finite element spaces (for velocity and pressure)
V = VectorFunctionSpace(mesh, "Lagrange", 1)
Q = FunctionSpace(mesh, "Lagrange", 1)

# Define trial and test functions 
u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)

# %% ====================================================================

# Define boundary conditions 
class DirichletBoundaryLower(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0)

class DirichletBoundaryUpper(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], H)

class DirichletBoundaryLeft(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0) 

class DirichletBoundaryRight(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], L)

class DirichletBoundaryObjects(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (not near(x[0], 0.0)) and (not near(x[0], L)) and (not near(x[1], 0.0)) and (not near(x[1], H))

dbc_lower = DirichletBoundaryLower()
dbc_upper = DirichletBoundaryUpper()
dbc_left = DirichletBoundaryLeft()
dbc_right = DirichletBoundaryRight()
dbc_objects = DirichletBoundaryObjects()

# Examples of time dependent and stationary inflow conditions
#uin = Expression('4.0*x[1]*(1-x[1])', element = V.sub(0).ufl_element())
#uin = Expression('1.0 + 1.0*fabs(sin(t))', element = V.sub(0).ufl_element(), t=0.0)
uin = 1.0
bcu_in0 = DirichletBC(V.sub(0), uin, dbc_left)
bcu_in1 = DirichletBC(V.sub(1), 0.0, dbc_left)
bcu_upp0 = DirichletBC(V.sub(0), 0.0, dbc_upper)
bcu_upp1 = DirichletBC(V.sub(1), 0.0, dbc_upper)
bcu_low0 = DirichletBC(V.sub(0), 0.0, dbc_lower)
bcu_low1 = DirichletBC(V.sub(1), 0.0, dbc_lower)
bcu_obj0 = DirichletBC(V.sub(0), 0.0, dbc_objects)
bcu_obj1 = DirichletBC(V.sub(1), 0.0, dbc_objects)

pin = Expression('5.0*fabs(sin(t))', element = Q.ufl_element(), t=0.0)
pout = 0.0
#bcp0 = DirichletBC(Q, pin, dbc_left) 
bcp1 = DirichletBC(Q, pout, dbc_right)

#bcu = [bcu_in0, bcu_in1, bcu_upp0, bcu_upp1, bcu_low0, bcu_low1, bcu_obj0, bcu_obj1]
bcu = [bcu_in0, bcu_in1, bcu_upp1, bcu_low1, bcu_obj0, bcu_obj1]
bcp = [bcp1]

# Define measure for boundary integration  
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# %% ====================================================================

nu = 4.0e-3

# %% ====================================================================

# Define iteration functions
# (u0,p0) solution from previous time step
# (u1,p1) linearized solution at present time step  
u0 = Function(V)
u1 = Function(V)
p0 = Function(Q)
p1 = Function(Q)
t = 0.

# Set parameters for nonlinear and lienar solvers 
num_nnlin_iter = 5 
prec = "amg" if has_krylov_solver_preconditioner("amg") else "default" 

# Time step length 
dt = 0.5*mesh.hmin()

# Define mesh deformation w, mesh velocity = w/dt
if meshdeform is None:
    w = Expression(("0.0", "0.0"), t=t, element = V.ufl_element())
else:
    #w = Expression((
    #    "amp_x*sin(2.0*pi*t*freq)*sin(pi*x[0]/L) - amp_x*sin(2.0*pi*(t-dt)*freq)*sin(pi*x[0]/L)",
    #    "amp_y*sin(2.0*pi*t*freq-0.5*pi)*sin(pi*x[1]/H) - amp_y*sin(2.0*pi*(t-dt)*freq-0.5*pi)*sin(pi*x[1]/H)"), L=L, H=H, t=t, dt=dt, amp_x=amp_x, amp_y=amp_y, freq=freq, element = V.ufl_element()) #make the displacement independant from dt

    [E, kwargs] = meshdeform
    w = Expression(E, t=t, dt=dt, **kwargs, element = V.ufl_element())

# %% ====================================================================

# Define variational problem

# Stabilization parameters
h = CellDiameter(mesh);
u_mag = sqrt(dot(u1,u1))
d1 = 1.0/sqrt((pow(1.0/dt,2.0) + pow(u_mag/h,2.0)))
d2 = h*u_mag

# Mean velocities for trapozoidal time stepping
um = 0.5*(u + u0)
um1 = 0.5*(u1 + u0)

# Momentum variational equation on residual form
Fu = inner((u - u0)/dt + grad(um)*(um1-w/dt), v)*dx - p1*div(v)*dx + nu*inner(grad(um), grad(v))*dx \
    + d1*inner((u - u0)/dt + grad(um)*(um1-w/dt) + grad(p1), grad(v)*(um1-w/dt))*dx + d2*div(um)*div(v)*dx 

au = lhs(Fu)
Lu = rhs(Fu)

# Continuity variational equation on residual form
Fp = d1*inner((u1 - u0)/dt + grad(um1)*(um1-w/dt) + grad(p), grad(q))*dx + div(um1)*q*dx 
ap = lhs(Fp)
Lp = rhs(Fp)

# %% ====================================================================
if psi is None:
    # Define the direction of the force to be computed in y
    psi_y_expression = Expression(("near(pow(x[0]-xc,2.0) + pow(x[1]-yc,2.0) - pow(rc,2.0), 0.0) ? phi_x : 0.","near(pow(x[0]-xc,2.0) + pow(x[1]-yc,2.0) - pow(rc,2.0), 0.0) ? phi_y : 0."), xc=xc, yc=yc, rc=rc, phi_x=0.0, phi_y=1.0, element = V.ufl_element())
    psi_y = interpolate(psi_y_expression, V)

    # Define the direction of the force to be computed in x
    psi_x_expression = Expression(("near(pow(x[0]-xc,2.0) + pow(x[1]-yc,2.0) - pow(rc,2.0), 0.0) ? phi_x : 0.","near(pow(x[0]-xc,2.0) + pow(x[1]-yc,2.0) - pow(rc,2.0), 0.0) ? phi_y : 0."), xc=xc, yc=yc, rc=rc, phi_x=1.0, phi_y=0.0, element = V.ufl_element())
    psi_x = interpolate(psi_x_expression, V)
else:
    psi_x, psi_y = psi
Forcey = inner((u1 - u0)/dt + grad(um1)*um1, psi_y)*dx - p1*div(psi_y)*dx + nu*inner(grad(um1), grad(psi_y))*dx
Forcex = inner((u1 - u0)/dt + grad(um1)*um1, psi_x)*dx - p1*div(psi_x)*dx + nu*inner(grad(um1), grad(psi_x))*dx

n = FacetNormal(mesh)
ds = Measure('ds', domain=mesh)
Forcex = dot(psi_x, nu*dot(grad(um1), n) - p1*n) * ds




#plt.figure()
#plot(psi, title="weight function psi")

# Force normalization
D = 2*rc
normalization = -2.0/D

# %% ====================================================================

# Force computation data 
force_x_array = np.array(0.0)
force_x_array = np.delete(force_x_array, 0)
force_y_array = np.array(0.0)
force_y_array = np.delete(force_y_array, 0)
time = np.array(0.0)
time = np.delete(time, 0)

print("Computing:")
print("[%-40s] %2.0f%s" % ("", 0, "%"), end="")

# %% ====================================================================


# Time stepping 
if uinit is not None:
    u0.assign(uinit)
    u1.assign(uinit)
if pinit is not None:
    p0.assign(pinit)
    p1.assign(pinit)

T = Tmax + tinit
t = dt + tinit


while t < T + DOLFIN_EPS:

    #s = 'Time t = ' + repr(t) 
    #print(s)

    pin.t = t
    #uin.t = t

    w.t = t
    ALE.move(mesh, w)

    # Solve non-linear problem 
    k = 0
    while k < num_nnlin_iter: 
        
        # Assemble momentum matrix and vector 
        Au = assemble(au)
        bu = assemble(Lu)

        # Compute velocity solution 
        [bc.apply(Au, bu) for bc in bcu]
        [bc.apply(u1.vector()) for bc in bcu]
        solve(Au, u1.vector(), bu, "bicgstab", "default")

        # Assemble continuity matrix and vector
        Ap = assemble(ap) 
        bp = assemble(Lp)

        # Compute pressure solution 
        [bc.apply(Ap, bp) for bc in bcp]
        [bc.apply(p1.vector()) for bc in bcp]
        solve(Ap, p1.vector(), bp, "bicgstab", prec)
        k += 1

    # Compute force
    Fx = assemble(Forcex)
    Fy = assemble(Forcey)
    force_x_array = np.append(force_x_array, normalization*Fx)
    force_y_array = np.append(force_y_array, normalization*Fy)
    time = np.append(time, t)

    # Update time step
    print("\r[%-40s] %2.0f%s" % ("="*int(np.ceil(40*(t-tinit)/Tmax)), 100*(t-tinit)/Tmax, "%"), end="")
    u0.assign(u1)
    t += dt

print("\r[%-40s] %2.0f%s" % ("="*40, 100, "%"))
drag = force_x_array
lift = force_y_array

#return u1, p1, time, drag, lift, dt, [psi_x, psi_y]


#%%
from dolfin import *

mesh = UnitCubeMesh(4,4,4)
F = FunctionSpace(mesh, "CG", 2)
V = VectorFunctionSpace(mesh, "CG", 2, dim=3)
E = Expression("x[0]*x[1]*x[2]*sin(x[0]*x[1]*x[2])", element = V.ufl_element())
u = interpolate(E, F)
grad_u = project(grad(u), V)

# Create subdomain (x0 = 1)
class Plane(SubDomain):
  def inside(self, x, on_boundary):
    return x[0] > 1.0 - DOLFIN_EPS

# Mark facets
facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
Plane().mark(facets, 1)
ds = Measure("ds")[facets]

### First method ###
# Define facet normal vector (built-in method)
n = FacetNormal(mesh)
flux_1 = assemble(dot(grad_u, n)*ds(1))

### Second method ###
# Manually define the normal vector
n = Constant((1.0,0.0,0.0))
flux_2 = assemble(dot(grad_u, n)*ds(1))

print("flux 1: ", flux_1)
print("flux 2: ", flux_2)
# %%
