
#%%
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from time import perf_counter as cl
# from line_profiler import LineProfiler
from numba import njit 
import os

import dolfin as dl
from dolfin.cpp.mesh import MeshQuality
from fenicstools.Interpolation import interpolate_nonmatching_mesh
dl.parameters['allow_extrapolation'] = True

import gmsh
import mshr

#%%
rotate_matrix = lambda t : np.moveaxis(np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]), [0, 1], [-2, -1])
defsize = np.array([6.4, 4.8])

#%%
@njit(cache=True)
def my_f(x, a=0, b=1):
	xsi = (x-b)/(a-b)
	return xsi**2 * (3-2*xsi)
@njit(cache=True)
def dmy_f(x, a=0, b=1):
	xsi = (x-b)/(a-b)
	return 6/(a-b)*(xsi - xsi**2)
@njit(cache=True)
def my_g(x, a=0, b=1):
	xsi = (x-b)/(a-b)
	return (x-a) * xsi**2
@njit(cache=True)
def dmy_g(x, a=0, b=1):
	xsi = (x-b)/(a-b)
	return xsi**2 + 2*(x-a)*xsi/(a-b)

@njit(cache=True)
def interpol_scalar(x, pts, v, dv):
	i = np.searchsorted(pts, x)
	n = len(pts)
	if i>0 and i < n:
		y = (my_f(x, pts[i], pts[i-1])*v[i] + my_f(x, pts[i-1], pts[i])*v[i-1])
		y += (my_g(x, pts[i], pts[i-1])*dv[i] + my_g(x, pts[i-1], pts[i])*dv[i-1])
		return y
	elif i==0:
		return v[0] + dv[0]*(x-pts[0])
	else:
		return v[-1] + dv[-1]*(x-pts[-1])


BoatMass = 224 * 1e6 #kg = 224000 tons
L = 400
U = 6
Mu = 1e-3
Rho = 1000
Nu = Mu/Rho # mu =1e-3, rho = 1000

Re = U*L/Nu
nu = 1/Re

L_c = 14 #depth of the boat
F0 = Mu * U * L_c * Re
M_0 = Rho*L*L*L_c

TINIT = 404.7372693860113
TINIT = 200

#%%
#%%


def interpol(x, pts, v, dv, deriv=False):
	if type(x) != np.ndarray:
		if not deriv:
			return interpol_scalar(x, pts, v, dv)
		x = np.array([x])
	if not deriv:
		y = np.zeros(x.shape)
		for i in range(len(pts)-1):
			y += (my_f(x, pts[i], pts[i+1])*v[i] + my_f(x, pts[i+1], pts[i])*v[i+1]) * (x >= pts[i]) * (x < pts[i+1])
			y += (my_g(x, pts[i], pts[i+1])*dv[i] + my_g(x, pts[i+1], pts[i])*dv[i+1]) * (x >= pts[i]) * (x < pts[i+1])
		y += v[0] * (x < pts[0])
		y += dv[0]*(x-pts[0]) * (x < pts[0])
		y += v[-1] * (x >= pts[-1])
		y += dv[-1]*(x-pts[-1]) * (x >= pts[-1])
	else:
		y = np.zeros(x.shape)
		for i in range(len(pts)-1):
			y += (dmy_f(x, pts[i], pts[i+1])*v[i] + dmy_f(x, pts[i+1], pts[i])*v[i+1]) * (x >= pts[i]) * (x < pts[i+1])
			y += (dmy_g(x, pts[i], pts[i+1])*dv[i] + dmy_g(x, pts[i+1], pts[i])*dv[i+1]) * (x >= pts[i]) * (x < pts[i+1])
		y += dv[0] * (x < pts[0])
		y += dv[-1] * (x >= pts[-1])
	
	if x.size == 1:
		y = y[0]
	return y




def baseval(x, i, pts, deriv=False):
	y = np.zeros(x.shape)
	if not deriv:
		if i==0:
			y += x < pts[0]
			y += my_f(x, pts[i], pts[i+1]) * (x >= pts[i]) * (x <= pts[i+1])

		elif i==len(pts)-1:
			y += my_f(x, pts[i], pts[i-1]) * (x >= pts[i-1]) * (x < pts[i])
			y += x >= pts[-1]

		else:
			y += my_f(x, pts[i], pts[i-1]) * (x >= pts[i-1]) * (x < pts[i])
			y += my_f(x, pts[i], pts[i+1]) * (x >= pts[i]) * (x <= pts[i+1])
		return y
	else:
		if i==0:
			y += dmy_f(x, pts[i], pts[i+1]) * (x >= pts[i]) * (x <= pts[i+1])

		elif i==len(pts)-1:
			y += dmy_f(x, pts[i], pts[i-1]) * (x >= pts[i-1]) * (x < pts[i])

		else:
			y += dmy_f(x, pts[i], pts[i-1]) * (x >= pts[i-1]) * (x < pts[i])
			y += dmy_f(x, pts[i], pts[i+1]) * (x >= pts[i]) * (x <= pts[i+1])
		return y


def baseder(x, i, pts, deriv=False):
	y = np.zeros(x.shape)
	if not deriv:
		if i==0:
			y += (x - pts[0]) * (x < pts[0])
			y += my_g(x, pts[i], pts[i+1]) * (x >= pts[i]) * (x <= pts[i+1])

		elif i==len(pts)-1:
			y += my_g(x, pts[i], pts[i-1]) * (x >= pts[i-1]) * (x < pts[i])
			y += (x - pts[-1]) * (x >= pts[-1])

		else:
			y += my_g(x, pts[i], pts[i-1]) * (x >= pts[i-1]) * (x < pts[i])
			y += my_g(x, pts[i], pts[i+1]) * (x >= pts[i]) * (x <= pts[i+1])
		return y
	else:
		if i==0:
			y += (x < pts[0])
			y += dmy_g(x, pts[i], pts[i+1]) * (x >= pts[i]) * (x <= pts[i+1])

		elif i==len(pts)-1:
			y += dmy_g(x, pts[i], pts[i-1]) * (x >= pts[i-1]) * (x < pts[i])
			y += (x >= pts[-1])

		else:
			y += dmy_g(x, pts[i], pts[i-1]) * (x >= pts[i-1]) * (x < pts[i])
			y += dmy_g(x, pts[i], pts[i+1]) * (x >= pts[i]) * (x <= pts[i+1])
		return y


smooth_coast = np.loadtxt("raw data/smooth_coast.csv", delimiter=',')
def lcoastfun(ystar):
	global L, U, Nu
	y = ystar * L
	[Ycoast, Left, dLeft, Righrt, dRighrt] = smooth_coast.T
	return interpol(y, Ycoast, Left, dLeft)/L

def rcoastfun(ystar):
	global L, U, Nu
	y = ystar * L
	[Ycoast, Left, dLeft, Righrt, dRighrt] = smooth_coast.T
	return interpol(y, Ycoast, Righrt, dRighrt)/L

smoothpath = np.loadtxt("raw data/smoothpath.csv", delimiter=',')
def posfun(tstar):
	t = tstar * L/U
	[T, X, dX, Y, dY] = smoothpath.T
	x = interpol(t+TINIT, T, X, dX)
	y = interpol(t+TINIT, T, Y, dY)
	return np.array([x, y])/L


def speedfun(tstar):
	t = tstar * L/U
	[T, X, dX, Y, dY] = smoothpath.T
	x = interpol(t+TINIT, T, X, dX, deriv=True)
	y = interpol(t+TINIT, T, Y, dY, deriv=True)
	return np.array([x, y])/U


smooth_hdg = np.loadtxt("raw data/smooth_hdg.csv", delimiter=',')
def hdgfun(tstar):
	t = tstar * L/U
	[TH, H, dH] = smooth_hdg.T
	return interpol(t+TINIT, TH, H, dH)

def hdgfunscal(tstar):
	t = tstar * L/U
	[TH, H, dH] = smooth_hdg.T
	return interpol_scalar(t+TINIT, TH, H, dH)

refboat = np.array([[-59/2, 59/2, 59/2, 0, -59/2, -59/2], [-202, -202, 140, 200, 140, -202]]) - [[0], [80]]
refboat = refboat / L

def boatfun(tstar):
	hdg = hdgfun(tstar)
	x, y = posfun(tstar)
	return rotate_matrix(-hdg).dot(refboat) + [[x], [y]]

# %%


def area(poly):
	x = poly[0]
	y = poly[1]
	n = len(x)

	A = 0
	for i in range(n-1):
		A += x[i]*y[i+1] - x[i+1]*y[i]
	A /= 2
	return A

def center_of_mass(poly):
	x = poly[0]
	y = poly[1]
	n = len(x)

	A = area(poly)
	c = np.zeros(2)
	for i in range(n-1):
		c[0] += (x[i]+x[i+1])*(x[i]*y[i+1] - x[i+1]*y[i])
		c[1] += (y[i]+y[i+1])*(x[i]*y[i+1] - x[i+1]*y[i])
	
	c /= 6*A
	return c

com_refboat = center_of_mass(refboat)
def pos_com(tstar):
	xy = posfun(tstar)
	theta = hdgfun(tstar)
	return xy + (rotate_matrix(-theta).dot(com_refboat)).T


def boatfun2(x, y, hdg):
	return rotate_matrix(-hdg).dot(refboat - com_refboat[:,None]) + [[x], [y]]

def compute_inertia():
	Iboat = refboat - com_refboat[:,None]
	vertecies = [dl.Point(x, y) for (x, y) in Iboat.T]
	Iboatpoly = mshr.Polygon(vertecies)
	mesh = mshr.generate_mesh(Iboatpoly, 1)

	f = dl.Expression("x[0]*x[0] + x[1]*x[1]", degree=2)
	V = dl.FunctionSpace(mesh, "Lagrange", 2)
	u = dl.project(f, V)
	I = dl.assemble(u*dl.dx) / area(Iboat)
	return I

#%%


def find_nex_pts(px, py, xnew):
	x0, x1, x2 = px
	y0, y1, y2 = py
	a1 = (xnew-x1)/(x0 - x1)
	a2 = (xnew-x2)/(x1 - x2)
	ny1 = a1*y0 + (1-a1)*y1
	ny2 = a2*y1 + (1-a2)*y2
	return [x0, xnew, xnew, x2], [y0, ny1, ny2, y2]

def find_intersect(boat, pos):
	P0 = boat[:,:-1][:,pos-1]
	P1 = boat[:,:-1][:,pos]
	P2 = boat[:,:-1][:,pos+1]
	P3 = boat[:,:-1][:,pos+2]
	xmed = (P1[0] + P2[0])/2
	A = np.array([P1-P0, P3-P2]).T
	ac = np.linalg.solve(A, P3-P0)
	NP = ac[0]*A[:,0] + P0
	px = [P0[0], NP[0], P3[0]]
	py = [P0[1], NP[1], P3[1]]
	return px, py, xmed

def updated_boat_geo(boat, xmin, xmax, ref):
	boat = np.copy(boat)
	lc = 2/ref #1e-2
	th = 1.5*lc

	N = boat.shape[1]
	xb = boat[0]
	yb = boat[1]

	flagged = xb[:-1] > xmax - th
	pos = np.nonzero(flagged)[0]
	NR = len(pos)
	if NR == 1:
		if xb[pos] < xmax:
			pos = pos[0]
			_, [y0, ny1, ny2, y2] = find_nex_pts(xb[:-1][[pos-1, pos, (pos+1)%N]], yb[:-1][[pos-1, pos, (pos+1)%N]], xmax-th)
			
			xb[pos] = xmax-th
			yb[pos] = ny1
			xb = np.insert(xb, pos+1, xmax-th)
			yb = np.insert(yb, pos+1, ny2)
		else:
			NR = -1

	
	elif NR == 2:
		pos = pos[0]
		px, py , xmed = find_intersect(boat, pos)
		xnew = xmax if xmed > (xmax - th/2) else xmax-th
		NR = -NR if xmed > (xmax - th/2) else NR

		_, [y0, ny1, ny2, y2] = find_nex_pts(px, py, xnew)
		xb[pos:pos+2] = xnew
		yb[pos:pos+2] = [ny1, ny2]
	elif NR > 2:
		print("THERE WAS AN ERROR IN THE MESH GENERATION")
	boat = np.vstack([xb, yb])
	boat[:,-1] = boat[:,0]

	flagged = xb[:-1] < xmin + th
	pos = np.nonzero(flagged)[0]
	NL = len(pos)
	if NL == 1:
		if xb[pos] > xmin:
			pos = pos[0]
			_, [y0, ny1, ny2, y2] = find_nex_pts(xb[:-1][[pos-1, pos, (pos+1)%N]], yb[:-1][[pos-1, pos, (pos+1)%N]], xmin+th)
			
			xb[pos] = xmin+th
			yb[pos] = ny1
			xb = np.insert(xb, pos+1, xmin+th)
			yb = np.insert(yb, pos+1, ny2)
		else:
			NL = -1

	elif NL == 2:
		pos = pos[0]
		px, py , xmed = find_intersect(boat, pos)
		xnew = xmin if xmed < (xmin + th/2) else xmin+th
		NL = -NL if xmed < (xmin + th/2) else NL

		_, [y0, ny1, ny2, y2] = find_nex_pts(px, py, xnew)
		xb[pos:pos+2] = xnew
		yb[pos:pos+2] = [ny1, ny2]
	elif NL > 2:
		print("THERE WAS AN ERROR IN THE MESH GENERATION")
	boat = np.vstack([xb, yb])
	boat[:,-1] = boat[:,0]

	return boat, (NL, NR)

def gmshGenMesh(boat, xmin, xmax, ymin, ymax, ref):
	gmsh.initialize()
	lc = 2/ref #1e-2
	th = 1.5*lc

	boat, meshtype = updated_boat_geo(boat, xmin, xmax, ref)

	points = []
	for i in range(boat.shape[1]-1):
		x, y = boat[:,i]
		points += [gmsh.model.occ.addPoint(x, y, 0)]

	lines = []
	lines2 = []
	for i in range(-1, boat.shape[1]-2):
		lines += [gmsh.model.occ.addLine(points[i], points[i+1])]
		lines2 += [gmsh.model.occ.addLine(points[i], points[i+1])]
	


	boat = gmsh.model.occ.addCurveLoop(lines)

	domain = gmsh.model.occ.addRectangle(xmin, ymin, 0, xmax-xmin, ymax-ymin)
	boat = gmsh.model.occ.addPlaneSurface([boat])
	stuff = gmsh.model.occ.cut([(2, domain)], [(2, boat)])
	final_geo_tags = [p[1] for p in stuff[0]]

	gmsh.model.occ.synchronize()

	# Local mesh size: fine near teh boat, coarser further away
	dist_field = gmsh.model.mesh.field.add("Distance")
	gmsh.model.mesh.field.setNumbers(dist_field, "CurvesList", lines2)
	gmsh.model.mesh.field.setNumber(dist_field, "NumPointsPerCurve", 5)

	ref_field = gmsh.model.mesh.field.add("Threshold")
	gmsh.model.mesh.field.setNumber(ref_field, "InField", dist_field)
	gmsh.model.mesh.field.setNumber(ref_field, "SizeMin", lc)
	gmsh.model.mesh.field.setNumber(ref_field, "SizeMax", lc*4)
	gmsh.model.mesh.field.setNumber(ref_field, "DistMin", 0.1)
	gmsh.model.mesh.field.setNumber(ref_field, "DistMax", 1.3)
	gmsh.model.mesh.field.setAsBackgroundMesh(ref_field)


	ps = gmsh.model.addPhysicalGroup(2, final_geo_tags)
	gmsh.model.setPhysicalName(2, ps, "My surface")

	# We can then generate a 2D mesh...
	gmsh.model.mesh.generate(2)

	# ... and save it to disk
	gmsh.write("mesh.msh2")
	gmsh.finalize()

	os.rename("mesh.msh2", "mesh.msh")
	os.system("dolfin-convert mesh.msh mesh.xml")
	mesh = dl.Mesh("mesh.xml")
	return mesh, meshtype

#%%


def compute_move_info(mesh, BCs):
	riverBC, coastBC, boatBC = BCs

	#### Mesh movement functions
	VE = dl.VectorElement("CG", mesh.ufl_cell(), 1)
	V = dl.FunctionSpace(mesh, VE)


	boatmovx = dl.Expression("cos(theta)*(x[0]-x0) - sin(theta)*(x[1]-y0) + x0 - x[0] + dx", theta=0.0, x0=0.0, y0=0.0, dx=0.0, element = V.ufl_element())
	boatmovy = dl.Expression("sin(theta)*(x[0]-x0) + cos(theta)*(x[1]-y0) + y0 - x[1] + dy", theta=0.0, x0=0.0, y0=0.0, dy=0.0, element = V.ufl_element())
	dom_move = dl.Expression("dy", dy=0.0, element = V.ufl_element())

	mybc0 = dl.DirichletBC(V.sub(0), boatmovx, boatBC)
	mybc1 = dl.DirichletBC(V.sub(1), boatmovy, boatBC)
	mybc2 = dl.DirichletBC(V.sub(0), 0, coastBC)
	#mybc3 = dl.DirichletBC(V.sub(1), dom_move, coastBC)
	#mybc4 = dl.DirichletBC(V.sub(0), 0, riverBC)
	mybc5 = dl.DirichletBC(V.sub(1), dom_move, riverBC)
	bcd = [mybc0, mybc1, mybc2, mybc5]

	u = dl.TrialFunction(V)
	v = dl.TestFunction(V) 
	d = dl.Function(V)

	f = dl.Expression(("0.0","0.0"), element = V.ufl_element())

	dim = u.geometric_dimension()
	E = 1
	nu = 0.2
	mu = E*0.5/(1+nu)
	lambda_ = nu*E/((1.0+nu)*(1.0-2.0*nu))

	def epsilon(u):
		return 0.5*(dl.grad(u) + dl.grad(u).T)

	def sigma(u):
		#return 2.0*mu*epsilon(u)
		return lambda_*dl.div(u)*dl.Identity(dim) + 2.0*mu*epsilon(u)

	# Define variational problem on residual form: r(u,v) = 0
	residual = ( dl.inner(sigma(u), epsilon(v))*dl.dx - dl.inner(f, v)*dl.dx )

	au = dl.lhs(residual)
	Lu = dl.rhs(residual)

	Ad = dl.assemble(au)
	bd = dl.assemble(Lu)

	[bc.apply(Ad, bd) for bc in bcd]
	[bc.apply(d.vector()) for bc in bcd]

	move_info = [Ad, bd, d, boatmovx, boatmovy, dom_move, bcd]
	return move_info


def remesh(boat, ref=20):
	x0, y0 = center_of_mass(boat)
	ymin, ymax = y0-800/L, y0+800/L
	xmin, xmax = lcoastfun(y0), rcoastfun(y0)

	# boat = boat[:,:-1]
	# vertecies = [dl.Point(x, y) for (x, y) in boat.T]
	# boatpoly = mshr.Polygon(vertecies)
	# river = mshr.Rectangle(dl.Point(xmin, ymin), dl.Point(xmax, ymax))
	# mesh = mshr.generate_mesh(river - boatpoly, ref)

	mesh, meshtype = gmshGenMesh(boat, xmin, xmax, ymin, ymax, ref)

	class RiverBC(dl.SubDomain):
		def inside(self, x, on_boundary):
			return on_boundary and (dl.near(x[1], ymin, eps=1e-6) or dl.near(x[1], ymax, eps=1e-6))

	class CoastBC(dl.SubDomain):
		def inside(self, x, on_boundary):
			return on_boundary and (dl.near(x[0], xmin, eps=1e-6) or dl.near(x[0], xmax, eps=1e-6))

	class BoatBC(dl.SubDomain):
		def inside(self, x, on_boundary):
			return on_boundary and not (dl.near(x[0], xmin, eps=1e-6) or dl.near(x[0], xmax, eps=1e-6) or dl.near(x[1], ymin, eps=1e-6) or dl.near(x[1], ymax, eps=1e-6))


	riverBC = RiverBC()
	coastBC = CoastBC()
	boatBC = BoatBC()
	BCs = [riverBC, coastBC, boatBC]
	boundary_marker = dl.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
	boundary_marker.set_all(0)
	coastBC.mark(boundary_marker, 1)
	boatBC.mark(boundary_marker, 2)

	#### Mesh movement functions
	move_info = compute_move_info(mesh, BCs)

	#### fluid dynamics
	flow_vars, flow_params = initiate_flow(mesh, BCs, move_info, boundary_marker)

	return mesh, meshtype, move_info, flow_vars, flow_params

def initiate_flow(mesh, BCs, move_info, boundary_marker):
	riverBC, coastBC, boatBC = BCs
	V = dl.VectorFunctionSpace(mesh, "Lagrange", 1)
	Q = dl.FunctionSpace(mesh, "Lagrange", 1)

	# Define trial and test functions 
	u = dl.TrialFunction(V)
	p = dl.TrialFunction(Q)
	v = dl.TestFunction(V)
	q = dl.TestFunction(Q)

	A, b, d, boatmovx, boatmovy, dom_move, bcd = move_info

	boatspeedx = dl.Expression("(cos(theta)*(x[0]-x0) - sin(theta)*(x[1]-y0) + x0 - x[0] + dx)/dt", theta=0.0, x0=0.0, y0=0.0, dx=0.0, dt=1.0, element = V.ufl_element())
	boatspeedy = dl.Expression("(sin(theta)*(x[0]-x0) + cos(theta)*(x[1]-y0) + y0 - x[1] + dy)/dt", theta=0.0, x0=0.0, y0=0.0, dy=0.0, dt=1.0, element = V.ufl_element())

	bcu0 = dl.DirichletBC(V.sub(0), boatspeedx, boatBC)
	bcu1 = dl.DirichletBC(V.sub(1), boatspeedy, boatBC)
	bcu2 = dl.DirichletBC(V.sub(0), 0.0, coastBC)
	bcu3 = dl.DirichletBC(V.sub(1), 0.0, coastBC)
	bcp0 = dl.DirichletBC(Q, 0.0, riverBC)

	bcu = [bcu0, bcu1, bcu2, bcu3]
	bcp = [bcp0]

	n = dl.FacetNormal(mesh)
	ds = dl.Measure('ds', domain=mesh, subdomain_data=boundary_marker)


	u0 = dl.Function(V)
	u1 = dl.Function(V)
	p1 = dl.Function(Q)

	# Time step length 
	hmin = mesh.hmin()
	mydt = dl.Expression("dt", dt=hmin, degree=0)

	nu = 1/Re
	h = dl.CellDiameter(mesh);
	u_mag = dl.sqrt(dl.dot(u1,u1))
	d1 = 2.0/dl.sqrt((pow(1.0/mydt,2.0) + pow(u_mag/h,2.0)))
	d2 = 2.0*h*u_mag

	# Mean velocities for trapozoidal time stepping
	um = 0.5*(u + u0)
	um1 = 0.5*(u1 + u0)

	# Momentum variational equation on residual form
	Fu = dl.inner((u - u0)/mydt + dl.grad(um)*(um1-d/mydt), v)*dl.dx - p1*dl.div(v)*dl.dx + nu*dl.inner(dl.grad(um), dl.grad(v))*dl.dx + d1*dl.inner((u - u0)/mydt + dl.grad(um)*(um1-d/mydt) + dl.grad(p1), dl.grad(v)*(um1-d/mydt))*dl.dx + d2*dl.div(um)*dl.div(v)*dl.dx 

	# Continuity variational equation on residual form
	Fp = d1*dl.inner((u1 - u0)/mydt + dl.grad(um1)*(um1-d/mydt) + dl.grad(p), dl.grad(q))*dl.dx + dl.div(um1)*q*dl.dx 

	au = dl.lhs(Fu)
	Lu = dl.rhs(Fu)
	ap = dl.lhs(Fp)
	Lp = dl.rhs(Fp)

	Rvec = dl.Expression(("x[0] - xc","x[1] - yc"), xc=0.0, yc=0.0, element=V.ufl_element())

	forcevec =  - (nu*dl.dot(dl.grad(u1), n) - p1*n)
	Forcex = forcevec[0] * ds(2)
	Forcey = forcevec[1] * ds(2)
	Torque = (Rvec[0]*forcevec[1] - Rvec[1]*forcevec[0]) * ds(2)

	flow_vars = [u0, u1, p1]
	forces = [Forcex, Forcey, Torque, Rvec]
	flow_params = [au, Lu, ap, Lp, bcu, bcp, hmin, mydt, boatspeedx, boatspeedy, V, forces]
	return flow_vars, flow_params

# %%

def set_dom_move(x0, y0, dx, dy, dtheta, boatmovx, boatmovy, dom_move):
	boatmovx.x0 = x0; boatmovx.y0 = y0	
	boatmovy.x0 = x0; boatmovy.y0 = y0	
	boatmovx.dx=dx
	boatmovy.dy=dy
	dom_move.dy=dy
	boatmovx.theta=dtheta
	boatmovy.theta=dtheta


def set_speeds(x0, y0, dx, dy, dtheta, dt, boatspeedx, boatspeedy):
	boatspeedx.x0 = x0; boatspeedx.y0 = y0	
	boatspeedy.x0 = x0; boatspeedy.y0 = y0	
	boatspeedx.dx=dx
	boatspeedy.dy=dy
	boatspeedx.theta=dtheta
	boatspeedy.theta=dtheta
	boatspeedx.dt=dt
	boatspeedy.dt=dt





#class BoundaryExpression(UserExpression): def eval(self, value, x): value[0] = 1 if subdomain.inside(x) else 0 
