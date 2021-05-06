# %% ============================================

# Load neccessary modules.
try:
    from google.colab import files #not needed if notebook is run locally
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



# %% ============================================

# Define domain 
L = 4
H = 4

# Define circle
xc = 0.5
yc = 0.5*H
rc = 0.2

def genmesh(resolution, xc, yc, rc):
    # Generate mesh (examples with and without a hole in the mesh) 
    #mesh = RectangleMesh(Point(0.0, 0.0), Point(L, H), L*resolution, H*resolution)
    mesh = generate_mesh(Rectangle(Point(0.0,0.0), Point(L,H)) - Circle(Point(xc,yc),rc), resolution)
    return mesh

resolution = 32
mesh = genmesh(resolution, xc, yc, rc)

plt.figure()
plot(mesh)
plt.show()

# %% ============================================

def compute_sol(mesh, Porder, tol=1e-3, maxref=0):
    # Generate mixed finite element spaces (for primal velocity and pressure)
    VE = VectorElement("CG", mesh.ufl_cell(), Porder+1)
    QE = FiniteElement("CG", mesh.ufl_cell(), Porder)
    WE = VE * QE

    W = FunctionSpace(mesh, WE)
    V = FunctionSpace(mesh, VE)
    Q = FunctionSpace(mesh, QE)

    # Define trial and test functions
    w = Function(W)
    (u, p) = (as_vector((w[0],w[1])), w[2])
    (v, q) = TestFunctions(W) 

    # Generate mixed finite element spaces (for adjoint velocity and pressure)
    VEa = VectorElement("CG", mesh.ufl_cell(), Porder+2)
    QEa = FiniteElement("CG", mesh.ufl_cell(), Porder+1)
    WEa = VEa * QEa

    Wa = FunctionSpace(mesh, WEa)
    Va = FunctionSpace(mesh, VEa)
    Qa = FunctionSpace(mesh, QEa)

    # Define adjoint trial and test functions
    wa = Function(Wa)
    (phi, theta) = (as_vector((wa[0],wa[1])), wa[2])
    (va, qa) = TestFunctions(Wa)


    # %% ============================================

    # Examples of inflow and outflow conditions
    XMIN = 0.0; XMAX = L
    YMIN = 0.0; YMAX = H
    uin = Expression(("4*(x[1]*(YMAX-x[1]))/(YMAX*YMAX)", "0."), YMAX=YMAX, element = V.ufl_element()) 
    #pout = 0.0

    # Inflow boundary (ib), outflow boundary (ob), body boundary (bb) and wall boundary (wb)
    ib = Expression("near(x[0],XMIN) ? 1. : 0.", XMIN=XMIN, element = Q.ufl_element())
    ob = Expression("near(x[0],XMAX) ? 1. : 0.", XMAX=XMAX, element = Q.ufl_element()) 
    wb = Expression("near(x[1],YMIN) || near(x[1],YMAX) ? 1. : 0.", YMIN=YMIN, YMAX=YMAX, element = Q.ufl_element())
    bb = Expression("x[0] > XMIN + DOLFIN_EPS && x[0] < XMAX - DOLFIN_EPS && x[1] > YMIN + DOLFIN_EPS && x[1] < YMAX - DOLFIN_EPS ? 1. : 0.", XMIN=XMIN, XMAX=XMAX, YMIN=YMIN, YMAX=YMAX, element = Q.ufl_element())

    # %% ============================================

    # Set boundary penalty parameter gamma 
    h = CellDiameter(mesh)
    C = 1.0e3
    gamma = C/h

    # Set force in primal problem
    f = Expression(("0.0","0.0"), element = V.ufl_element())

    # Set data that describe functional that defines the adjoint problem
    #psi1 = Expression(("exp(-10.0*(pow(x[0]-2.0,2) + pow(x[1]-1.5,2)))","0.0"), element = V.ufl_element())
    #psi2 = Expression("exp(-10.0(pow(x[0]-2.0,2) + pow(x[1]-1.0,2)))", element = Q.ufl_element())
    psi1 = Expression(("0.0","0.0"), element = V.ufl_element())
    psi2 = Expression("0.0", element = Q.ufl_element())
    phi3 = Expression(("1.0","0.0"), element = V.ufl_element()) #boundary term

    # Define primal variational problem on residual form: r(u,p;v,q) = 0
    res = ( -p*div(v)*dx + inner(grad(u), grad(v))*dx + div(u)*q*dx - inner(f, v)*dx + 
            gamma*(ib*inner(u - uin, v) + wb*inner(u, v) + bb*inner(u, v))*ds )

    # Solve primal algebraic system 
    solve(res == 0, w) 

    # Define adjoint variational problem on residual form: r(u,p;v,q) = 0
    res_a = ( -qa*div(phi)*dx + inner(grad(va), grad(phi))*dx + div(va)*theta*dx + 
            gamma*(ib*inner(phi, va) + wb*inner(phi, va) + bb*inner(phi - phi3, va))*ds 
            - inner(va, psi1)*dx - qa*psi2*dx )

    # Solve adjoint algebraic system 
    solve(res_a == 0, wa)

    Force = inner(u, psi1)*dx + p*psi2*dx + inner(phi3, u)*ds
    Force = assemble(Force)

    err_ind_sum = ( inner(f, phi)*dx + p*div(phi)*dx - inner(grad(u), grad(phi))*dx - div(u)*theta*dx - 
                gamma*(ib*inner(u - uin, phi) + wb*inner(u, phi) + bb*inner(u, phi))*ds )
    tot_err = assemble(err_ind_sum)
    print("Force = %f, total error = %.3e" % (Force, np.abs(tot_err)))

    if maxref > 0 and np.abs(tot_err) > tol:
        # Define function space over the elements of the mesh
        WDG = FunctionSpace(W.mesh(), "DG", 0)
        elm = TestFunction(WDG)
        err_ind = Function(WDG)

        # Compute local error indicators over the cells of the mesh 
        local_error = ( elm*inner(f, phi)*dx + elm*p*div(phi)*dx - elm*inner(grad(u), grad(phi))*dx - elm*div(u)*theta*dx ) 
        err_ind.vector()[:] = assemble(local_error)
        err_ind_abs = np.abs(err_ind.vector())
        err_ind_mean = err_ind_abs.sum()/err_ind.vector().size()

        # Local mesh refinement (specified by a cell marker)
        no_levels = 1
        for i in range(0,no_levels):
            cell_marker = MeshFunction("bool", mesh, mesh.topology().dim())
            for c in cells(mesh):
                cell_marker[c] = False
                local_error_cell = err_ind_abs[c.index()] 
                if local_error_cell > err_ind_mean:
                    cell_marker[c] = True
        mesh = refine(mesh, cell_marker)

        return compute_sol(mesh, Porder, tol=tol, maxref=maxref-1)


    return u, p, phi, theta, V, Q, mesh



# %% ============================================

# %% ============================================

u, p, phi, theta, V, Q, mesh = compute_sol(mesh=mesh, Porder=1, tol = 1e-2, maxref=3)

# %% ============================================



# Plot solution
f2plot = [u, p, phi, theta]
titles = ["Velocity", "Pressure", "Adjoint velocity", "Adjoint pressure"]
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(fsize*L, fsize*H))
axes = axes.ravel()
for (i, fun) in enumerate(f2plot):
    plt.sca(axes[i])
    plot(mesh, color="k", linewidth=0.1, alpha=0.7)
    plot(fun, title=titles[i])

plt.show()

# Export files
#!tar -czvf results-Stokes.tar.gz results-NS
#files.download('results-Stokes.tar.gz')

# %% ============================================




# %% ============================================


psi1 = Expression(("0.0", "exp(-10.0*(pow(x[0]-2.0,2) + pow(x[1]-1.5,2)))"), element = V.ufl_element())

exp(-( pow((x[0]-1), 2.0) + pow((x[1]-2), 2.0))/(0.5*0.5) )
# %% ============================================
