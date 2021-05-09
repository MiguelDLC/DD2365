
#%%
import numpy as np
from matplotlib import pyplot as plt
R = 6371000

Lcoast = np.loadtxt("Lcoast.txt", delimiter=",")
Rcoast = np.loadtxt("Rcoast.txt", delimiter=",")
path = np.loadtxt("path.txt", delimiter=",")
latlong = path.mean(axis=0)
transform = np.array([[0, np.cos(np.deg2rad(latlong[0]))], [1, 0]])*R*np.pi/180

Lcoast = (Lcoast - latlong).dot(transform)
Rcoast = (Rcoast - latlong).dot(transform)
path = (path - latlong).dot(transform)

defsize = np.array([6.4, 4.8])
fig = plt.figure(figsize=2*defsize)
plt.plot(Lcoast[:,0], Lcoast[:,1], c="#1f77b4")
plt.plot(Rcoast[:,0], Rcoast[:,1], c="#1f77b4")
plt.plot(path[:,0], path[:,1], "k")
plt.ylim([-2000, 2000])
plt.gca().set_aspect(1)
plt.savefig("../path.pdf")

# %%




plt.plot(path)
# %%


def f(x, a=0, b=1):
	xsi = (x-b)/(a-b)
	return xsi**2 * (3-2*xsi)

def df(x, a=0, b=1):
	xsi = (x-b)/(a-b)
	return 6/(a-b)*(xsi - xsi**2)

def g(x, a=0, b=1):
	xsi = (x-b)/(a-b)
	return (x-a) * xsi**2

def dg(x, a=0, b=1):
	xsi = (x-b)/(a-b)
	return xsi**2 + 2*(x-a)*xsi/(a-b)

def interpol(x, pts, v, dv, deriv=False):
	if not deriv:
		y = np.zeros(x.shape)
		for i in range(len(pts)-1):
			y += (f(x, pts[i], pts[i+1])*v[i] + f(x, pts[i+1], pts[i])*v[i+1]) * (x >= pts[i]) * (x < pts[i+1])
			y += (g(x, pts[i], pts[i+1])*dv[i] + g(x, pts[i+1], pts[i])*dv[i+1]) * (x >= pts[i]) * (x < pts[i+1])
		y += v[0] * (x < pts[0])
		y += dv[0]*(x-pts[0]) * (x < pts[0])
		y += v[-1] * (x >= pts[-1])
		y += dv[-1]*(x-pts[-1]) * (x >= pts[-1])
	else:
		y = np.zeros(x.shape)
		for i in range(len(pts)-1):
			y += (df(x, pts[i], pts[i+1])*v[i] + df(x, pts[i+1], pts[i])*v[i+1]) * (x >= pts[i]) * (x < pts[i+1])
			y += (dg(x, pts[i], pts[i+1])*dv[i] + dg(x, pts[i+1], pts[i])*dv[i+1]) * (x >= pts[i]) * (x < pts[i+1])
		y += dv[0] * (x < pts[0])
		y += dv[-1] * (x >= pts[-1])
	return y

def baseval(x, i, pts, deriv=False):
	y = np.zeros(x.shape)
	if not deriv:
		if i==0:
			y += x < pts[0]
			y += f(x, pts[i], pts[i+1]) * (x >= pts[i]) * (x <= pts[i+1])

		elif i==len(pts)-1:
			y += f(x, pts[i], pts[i-1]) * (x >= pts[i-1]) * (x < pts[i])
			y += x >= pts[-1]

		else:
			y += f(x, pts[i], pts[i-1]) * (x >= pts[i-1]) * (x < pts[i])
			y += f(x, pts[i], pts[i+1]) * (x >= pts[i]) * (x <= pts[i+1])
		return y
	else:
		if i==0:
			y += df(x, pts[i], pts[i+1]) * (x >= pts[i]) * (x <= pts[i+1])

		elif i==len(pts)-1:
			y += df(x, pts[i], pts[i-1]) * (x >= pts[i-1]) * (x < pts[i])

		else:
			y += df(x, pts[i], pts[i-1]) * (x >= pts[i-1]) * (x < pts[i])
			y += df(x, pts[i], pts[i+1]) * (x >= pts[i]) * (x <= pts[i+1])
		return y


def baseder(x, i, pts, deriv=False):
	y = np.zeros(x.shape)
	if not deriv:
		if i==0:
			y += (x - pts[0]) * (x < pts[0])
			y += g(x, pts[i], pts[i+1]) * (x >= pts[i]) * (x <= pts[i+1])

		elif i==len(pts)-1:
			y += g(x, pts[i], pts[i-1]) * (x >= pts[i-1]) * (x < pts[i])
			y += (x - pts[-1]) * (x >= pts[-1])

		else:
			y += g(x, pts[i], pts[i-1]) * (x >= pts[i-1]) * (x < pts[i])
			y += g(x, pts[i], pts[i+1]) * (x >= pts[i]) * (x <= pts[i+1])
		return y
	else:
		if i==0:
			y += (x < pts[0])
			y += dg(x, pts[i], pts[i+1]) * (x >= pts[i]) * (x <= pts[i+1])

		elif i==len(pts)-1:
			y += dg(x, pts[i], pts[i-1]) * (x >= pts[i-1]) * (x < pts[i])
			y += (x >= pts[-1])

		else:
			y += dg(x, pts[i], pts[i-1]) * (x >= pts[i-1]) * (x < pts[i])
			y += dg(x, pts[i], pts[i+1]) * (x >= pts[i]) * (x <= pts[i+1])
		return y





pts = [0, 1, 2, 3, 4]
v = [1, 2, 3, 4, 5]
dv = [1, 1, 1, 1, 0.5]

a = 0
b = 5
x = np.linspace(a-1, b+1, 1001)
h = x[1] - x[0]

ddf = np.convolve(f(x, a, b), [0.5/h, 0, -0.5/h])[1:-1]
ddf[0] = 0
ddf[-1] = 0

#plt.plot(x, x, "k:")
#plt.plot(x, f(x, a, b))
#plt.plot(x, df(x, a, b))
#plt.plot(x, g(x, a, b))
#plt.plot(x, dg(x, a, b))
#plt.plot(x, interpol(x, pts, v, dv))
#plt.plot(x, interpol(x, pts, v, dv, deriv=True))

#plt.plot(x, baseval(x, 0, pts, deriv=False))
#plt.plot(x, baseval(x, 0, pts, deriv=True))

plt.plot(x, baseder(x, 4, pts, deriv=False))
plt.plot(x, baseder(x, 4, pts, deriv=True))


def residual(XdXYdY, T, t, path):
	np = len(T)
	XdXYdY = XdXYdY.reshape((n, 4))
	X  = XdXYdY[:,0]
	dX = XdXYdY[:,1]
	Y  = XdXYdY[:,2]
	dY = XdXYdY[:,3]

	fx = interpol(t, T, X, dX)
	fy = interpol(t, T, Y, dY)

	x = path[:,0]
	y = path[:,1]
	rx = x - fx
	ry = y - fy

	return rx.dot(rx) + ry.dot(ry)




# %%
