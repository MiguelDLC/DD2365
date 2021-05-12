
#%%
import parse
import codecs
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.optimize import minimize
from time import perf_counter as cl
from line_profiler import LineProfiler
from numba import njit 

rotate_matrix = lambda t : np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
defsize = np.array([6.4, 4.8])

#%%
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
def lcoastfun(y):
	[Ycoast, L, dL, R, dR] = smooth_coast.T
	return interpol(y, Ycoast, L, dL)

def rcoastfun(y):
	[Ycoast, L, dL, R, dR] = smooth_coast.T
	return interpol(y, Ycoast, R, dR)

smoothpath = np.loadtxt("raw data/smoothpath.csv", delimiter=',')
def posfun(t):
	[T, X, dX, Y, dY] = smoothpath.T
	x = interpol(t+404.7372693860113, T, X, dX)
	y = interpol(t+404.7372693860113, T, Y, dY)
	return np.array([x, y])

smooth_hdg = np.loadtxt("raw data/smooth_hdg.csv", delimiter=',')
def hdgfun(t):
	[TH, H, dH] = smooth_hdg.T
	return interpol(t+404.7372693860113, TH, H, dH)

def hdgfunscal(t):
	[TH, H, dH] = smooth_hdg.T
	return interpol_scalar(t+404.7372693860113, TH, H, dH)

refboat = np.array([[-59/2, 59/2, 59/2, 0, -59/2, -59/2], [-200, -200, 150, 200, 150, -200]]) - [[0], [80]]

def boatfun(t):
	hdg = hdgfun(t)
	x, y = posfun(t)
	return rotate_matrix(-hdg).dot(refboat) + [[x], [y]]

# %%

import dolfin as dl
import mshr
from dolfin.cpp.mesh import MeshQuality

t0 = 450
L = 400

#%%


tees = np.linspace(0, 450, 10)
for i, loc_time in enumerate(tees):
	
	y0 = posfun(loc_time)[1]
	boat = boatfun(loc_time)
	vertecies = [dl.Point(x, y) for (x, y) in boat.T]
	mesh = mshr.generate_mesh(mshr.Rectangle(dl.Point(lcoastfun(y0-500),y0-500), dl.Point(rcoastfun(y0+500),y0+500)) - mshr.Polygon(vertecies), 20)


	fig = plt.figure(figsize=defsize*[2,3])
	dl.plot(mesh)
	plt.gca().set_aspect(1)
	plt.show()




#%%


def remesh(t):
	x0, y0 = posfun(t)
	boat = boatfun(t)
	vertecies = [dl.Point(x, y) for (x, y) in boat.T]
	mesh = mshr.generate_mesh(mshr.Rectangle(dl.Point(lcoastfun(y0-500),y0-500), dl.Point(rcoastfun(y0+500),y0+500)) - mshr.Polygon(vertecies), 20)


	class CoastBC(dl.SubDomain):
		def inside(self, x, on_boundary):
			return on_boundary and (dl.near(x[0], lcoastfun(y0-500)) or dl.near(x[0], rcoastfun(y0+500)) or dl.near(x[1], y0-500) or dl.near(x[1], y0+500))

	class BoatBC(dl.SubDomain):
		def inside(self, x, on_boundary):
			return on_boundary and not (dl.near(x[0], lcoastfun(y0-500)) or dl.near(x[0], rcoastfun(y0+500)) or dl.near(x[1], y0-500) or dl.near(x[1], y0+500))

	coastBC = CoastBC()
	boatBC = BoatBC()

	VE = dl.VectorElement("CG", mesh.ufl_cell(), 1)
	V = dl.FunctionSpace(mesh, VE)


	movx = dl.Expression("cos(theta)*(x[0]-x0) - sin(theta)*(x[1]-y0) + x0 - x[0] + dx", theta=0.0, x0=0.0, y0=0.0, dx=0.0, element = V.ufl_element())
	movy = dl.Expression("sin(theta)*(x[0]-x0) + cos(theta)*(x[1]-y0) + y0 - x[1] + dy", theta=0.0, x0=0.0, y0=0.0, dy=0.0, element = V.ufl_element())
	move = dl.Expression("dy", dy=0.0, element = V.ufl_element())

	mybc0 = dl.DirichletBC(V.sub(0), movx, boatBC)
	mybc1 = dl.DirichletBC(V.sub(1), movy, boatBC)
	mybc2 = dl.DirichletBC(V.sub(0), 0, coastBC)
	mybc3 = dl.DirichletBC(V.sub(1), move, coastBC)
	bcu = [mybc0, mybc1, mybc2, mybc3]

	u = dl.TrialFunction(V)
	v = dl.TestFunction(V) 
	d = dl.Function(V)

	f = dl.Expression(("0.0","0.0"), element = V.ufl_element())

	dim = u.geometric_dimension()
	E = 1.0e10
	nu = 0.3
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

	A = dl.assemble(au)
	b = dl.assemble(Lu)

	[bc.apply(A, b) for bc in bcu]
	[bc.apply(d.vector()) for bc in bcu]

	move_info = [A, b, d, movx, movy, move, bcu]
	return mesh, move_info


# %%

def meshmove(t0, t1, mesh,move_info):
	A, b, d, movx, movy, move, bcu = move_info
	x0, y0 = posfun(t0)
	x1, y1 = posfun(t1)

	dx = x1 - x0
	dy = y1 - y0

	theta = hdgfun(t0) - hdgfun(t1)
	movx.x0 = x0; movx.y0 = y0	
	movy.x0 = x0; movy.y0 = y0
	movx.dx=dx
	movy.dy=dy
	move.dy=dy
	movx.theta=theta
	movy.theta=theta

	[bc.apply(A, b) for bc in bcu]
	[bc.apply(d.vector()) for bc in bcu]
	dl.solve(A, d.vector(), b, "bicgstab", "default")
	dl.ALE.move(mesh, d)




#class BoundaryExpression(UserExpression): def eval(self, value, x): value[0] = 1 if subdomain.inside(x) else 0 
#%%
lp = LineProfiler()
lp_wrapper = lp(meshmove)

lp.add_function(posfun)
lp.add_function(hdgfun)
lp.add_function(interpol)
mesh, move_info = remesh(0.0)


fig = plt.figure(figsize=defsize*[2,3])
dl.plot(mesh,  linewidth=0.5)
plt.show()
MeshQuality.radius_ratio_min_max(mesh)

t = 0.0
dt = 2
i = 0
while t < 100:
	et = cl()
	lp_wrapper(t, t+dt, mesh, move_info)
	et = cl() - et
	print(2000*et)
	t += dt

	if (i % 100) ==0:
		fig = plt.figure(figsize=defsize*[2,3])
		dl.plot(mesh, linewidth=0.5)
		plt.savefig("Mesh/img%0.3d" % i)
		plt.show()
	

	if MeshQuality.radius_ratio_min_max(mesh)[0] < 0.15:
		mesh, move_info = remesh(t)
		#Funtion project
	
	i += 1

lp.print_stats()

# %%
