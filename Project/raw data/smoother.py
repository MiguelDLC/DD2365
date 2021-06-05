
#%%
import parse
import codecs
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.optimize import minimize


# %%

rotate_matrix = lambda t : np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

def my_f(x, a=0, b=1):
	xsi = (x-b)/(a-b)
	return xsi**2 * (3-2*xsi)

def dmy_f(x, a=0, b=1):
	xsi = (x-b)/(a-b)
	return 6/(a-b)*(xsi - xsi**2)

def my_g(x, a=0, b=1):
	xsi = (x-b)/(a-b)
	return (x-a) * xsi**2

def dmy_g(x, a=0, b=1):
	xsi = (x-b)/(a-b)
	return xsi**2 + 2*(x-a)*xsi/(a-b)

def interpol(x, pts, v, dv, deriv=False):
	if type(x) != np.ndarray:
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

def optimparam(t, T, data):
	n = len(T)
	base = []
	for i in range(n):
		base += [baseval(t, i, T), baseder(t, i, T)/20]
	base = base[:-1] #force last derivative to be 0

	A = np.array(base).T

	x = np.linalg.solve(A.T @ A, A.T @ data)
	x = np.hstack([x, [0.0]])
	x = x.reshape([n,2])
	x[:,1] /= 20
	return x

#%%
R = 6371000

Lcoast = np.loadtxt("Lcoast.txt", delimiter=",")
Rcoast = np.loadtxt("Rcoast.txt", delimiter=",")
path = np.loadtxt("path.txt", delimiter=",")
latlong = path.mean(axis=0)
transform = np.array([[0, np.cos(np.deg2rad(latlong[0]))], [1, 0]])*R*np.pi/180

Lcoast = (Lcoast - latlong).dot(transform)
Rcoast = (Rcoast - latlong).dot(transform)
path = (path - latlong).dot(transform)

a, b = np.polyfit(Lcoast[65:,1], Lcoast[65:,0], 1)
theta = np.arctan(a)
rotate = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

Lcoast = Lcoast.dot(rotate)
Rcoast = Rcoast.dot(rotate)
path = path.dot(rotate)

# %%

ly = Lcoast[:,1]
lx = Lcoast[:,0]
ry = Rcoast[:,1]
rx = Rcoast[:,0]

y = np.linspace(-3000, 3000, 1001)

l = interpolate.interp1d(ly, lx)(y)
r = interpolate.interp1d(ry, rx)(y)

Y = [-3000, -2250, -1500, -500]

LdL = optimparam(y, Y, l)
RdR = optimparam(y, Y, r)
L  = LdL[:,0]; dL = LdL[:,1]
R  = RdR[:,0]; dR = RdR[:,1]

xmax = np.array([L[-1], R[-1]])
d = (xmax[1] - xmax[0])
off = -xmax.mean()
scale_off = (d - 290)/2
Lcoast[:,0] += off
Rcoast[:,0] += off
path[:,0] += off
L += off + scale_off -2
R += off - scale_off -2

smooth_coast = np.vstack([Y, L, dL, R, dR]).T
np.savetxt("smooth_coast.csv", smooth_coast, delimiter=',', header="Y, Xleft, dXleft, Xright, dXright")

# %%
defsize = np.array([6.4, 4.8])
fig = plt.figure(figsize=2*defsize)
plt.plot(Lcoast[:,0], Lcoast[:,1], c="#1f77b4")
plt.plot(Rcoast[:,0], Rcoast[:,1], c="#1f77b4")
plt.plot(path[:,0], path[:,1], "k")
plt.ylim([1500, 2000])
plt.xlim([-250, 250])
plt.gca().set_aspect(1)
plt.tight_layout()
plt.savefig("../path.png")
plt.show()

# %%


def process_line(l):
	sog = np.NaN
	hdg = np.NaN
	cog = np.NaN
	rot = np.NaN
	draught = np.NaN
	t = np.NaN

	for w in l.split(","):
		w = w.replace("°", "")
		if w[0:3] == "SOG":
			try:
				sog = list(parse.parse("SOG: {:f} Knts", w))[0]
			except:
				pass

		if w[0:3] == "HDG":
			try:
				hdg = list(parse.parse("HDG: {:f}", w))[0]
			except:
				pass

		if w[0:3] == "COG":
			try:
				cog = list(parse.parse("COG: {:f}", w))[0]
			except:
				pass

		if w[0:3] == "ROT":
			try:
				rot = list(parse.parse("ROT: {:f}", w))[0]
			except:
				pass

		if w[0:7] == "Draught":
			try:
				draught = list(parse.parse("Draught: {:f} m", w))[0]
			except:
				pass

		if w[0:11] == "23 Mar 2021":
			try:
				h, m, s = list(parse.parse("23 Mar 2021 {:d}:{:d}:{:d} UTC", w))
				t = 3600*h + 60*m + s
			except:
				pass

	data = np.array([t, sog, hdg, cog, rot, draught])
	return data


def fill_nan(a):
	inds = np.arange(a.shape[0])
	good = np.where(np.isfinite(a))
	f = interpolate.interp1d(inds[good], a[good], bounds_error=False)
	return f(inds)



with codecs.open("Data.txt", "r", "utf-8") as f:
	data = f.read().split("\n")[:-1]
	data = np.array([process_line(l) for l in data])
	for i in range(data.shape[1]):
		data[:, i] = fill_nan(data[:, i])


def smooth(data, n=50):
	s = data
	for i in range(n):
		s = np.convolve(s, [0.25, 0.5, 0.25], mode="same")
		s[0] = data[0]
		s[-1] = data[-1]
	return s

y = data[:,0]
x = np.arange(len(y))
[c1, c0] = np.polyfit(x, y, 1)
data[:, 0] = c1*x; t = data[:,0]

#%%
[sog, hdg, cog, rot, draught] = [data[:, 1], data[:, 2], data[:, 3], data[:,4], data[:,5]]

cog = np.deg2rad((cog+250) % 360 - 250) - theta
hdg = np.deg2rad((hdg+250) % 360 - 250) - theta
sog *= 0.514444 #knots to m/s


#fig = plt.figure(figsize=2*defsize)
#plt.plot(data[:, 0], data[:, 1])


#%%


def optimparam(t, T, data):
	n = len(T)
	base = []
	for i in range(n):
		base += [baseval(t, i, T), baseder(t, i, T)/20]
	base = base[:-1] #force last derivative to be 0

	A = np.array(base).T

	x = np.linalg.solve(A.T @ A, A.T @ data)
	x = np.hstack([x, [0.0]])
	x = x.reshape([n,2])
	x[:,1] /= 20
	return x

bc = ((2, 0.0), (1, 0.0))
def err(XY, T, t, path):
	n = len(T)
	X = XY[:n]
	Y = XY[n:]
	funx = interpolate.CubicSpline(T, X, bc_type=bc)
	funy = interpolate.CubicSpline(T, Y, bc_type=bc)

	x = funx(t)
	y = funy(t)
	diffx = (x - path[:,0])
	diffy = (y - path[:,1])
	return diffx.dot(diffx) + diffy.dot(diffy)


#%%





T = np.array([0, 75, 125, 145, 170, 190, 225, 265, 300, 370, 390, 450, 525, 600, 675, 710, 720, 730, 820])
def cb(xk):
	print(err(xk, T, t, path))

ind = t.searchsorted(T)


XdX = optimparam(t, T, path[:,0])
YdY = optimparam(t, T, path[:,1])
X  = XdX[:,0]; dX = XdX[:,1]
Y  = YdY[:,0]; dY = YdY[:,1]


XY = np.hstack([X, Y])
print("Initial error:", err(XY.ravel(), T, t, path))

opt = minimize(err, XY, args=(T, t, path), tol=1.0)
XY = opt.x
XY = XY.reshape((2, -1))
X = XY[0]
Y = XY[1]

funx = interpolate.CubicSpline(T, X, bc_type=bc)
funy = interpolate.CubicSpline(T, Y, bc_type=bc)

dX = funx.derivative()(T)
dY = funy.derivative()(T)

print("Final error:", err(XY.ravel(), T, t, path))

smoothpath = np.vstack([T, X, dX, Y, dY]).T
np.savetxt("smoothpath.csv", smoothpath, delimiter=',', header="T, X, dX, Y, dY")
[T, X, dX, Y, dY] = smoothpath.T

plt.figure(figsize=(20, 10))
plt.plot(t, path[:,0])
plt.plot(t, interpol(t, T, X, dX))
plt.plot(T, interpol(T, T, X, dX), ".k")
plt.show()
plt.figure(figsize=(20, 10))
plt.plot(t, path[:,1])
plt.plot(t, interpol(t, T, Y, dY))
plt.plot(T, interpol(T, T, Y, dY), ".k")
plt.show()


#%%
# 
# fig = plt.figure(figsize=2*defsize)
# plt.plot(Lcoast[:,0], Lcoast[:,1], c="#1f77b4")
# plt.plot(Rcoast[:,0], Rcoast[:,1], c="#1f77b4")
# plt.plot(path[:,0], path[:,1], "k")
# plt.plot(path[ind,0], path[ind,1], "ok")
# plt.plot(X, Y, "*")
# plt.ylim([1600, 1900])
# plt.xlim([-250, 200])
# plt.gca().set_aspect(1)
# plt.plot(interpol(t, T, X, dX),interpol(t, T, Y, dY))
# plt.show()
# # %%
# 
# n = 100
# ker = np.exp(-np.linspace(-5, 5, n+1)**2); ker /= ker.sum()
# 
# def smooth(a):
# 	return np.convolve(a, ker, mode="same")
# 
# fig = plt.figure(figsize=2*defsize)
# plt.plot(t, path[:,0], linewidth=2)
# plt.plot(t, path[:,1], linewidth=2)
# 
# plt.plot(T, X, ".", c="k")
# plt.plot(t*1.1, interpol(t*1.1, T, X, dX), "k", linewidth=0.7)
# 
# plt.plot(T, Y, ".", c="k")
# plt.plot(t*1.1, interpol(t*1.1, T, Y, dY), "k", linewidth=0.7)
# #plt.xlim([500, 850])
# #plt.ylim(1400, 1800)
# #plt.ylim([-500, -100])
# plt.show()

# %%






TH = np.array([0, 15, 25, 50, 100, 150, 200, 250, 315, 360, 400, 420, 450, 500, 550, 600, 675, 700, 720, 735, 745, 760, 785, 790, 795])

HdH = optimparam(t, TH, hdg)
H  = HdH[:,0]; dH = HdH[:,1]




bc = ((2, 0.0), (1, 0.0))
def errhdg(H, T, t, hdg):
	n = len(T)
	funh = interpolate.CubicSpline(T, H, bc_type=bc)

	h = funh(t)
	diffh = (h - hdg)
	return diffh.dot(diffh)


print("Initial error HDG:", errhdg(H, TH, t, hdg))

opt = minimize(errhdg, H, args=(TH, t, hdg), tol=1.0e-3)
H = opt.x
print("Final error HDG:", errhdg(H, TH, t, hdg))

funh = interpolate.CubicSpline(TH, H, bc_type=bc)
dH = funh.derivative()(TH)




smooth_hdg = np.vstack([TH, H, dH]).T
np.savetxt("smooth_hdg.csv", smooth_hdg, delimiter=',', header="TH, H, dH")
[TH, H, dH] = smooth_hdg.T

plt.figure(figsize=(20, 10))
plt.plot(t, hdg)
plt.plot(t, interpol(t, TH, H, dH))
plt.plot(TH, interpol(TH, TH, H, dH), ".k")
plt.show()

# %%
# %%


def hdgfun(t):
	[TH, H, dH] = smooth_hdg.T
	return interpol(t, TH, H, dH)

def posfun(t):
	[T, X, dX, Y, dY] = smoothpath.T
	x = interpol(t, T, X, dX)
	y = interpol(t, T, Y, dY)
	return np.array([x, y])


x = [-59/2, 59/2, 59/2, 0, -59/2, -59/2]
y = np.array([-200, -200, 150, 200, 150, -200])-80
refboat = np.array([x, y])


def plotboat(x, y, hdg):
	lboat = rotate_matrix(-hdg).dot(refboat) + [[x], [y]]
	plt.fill(lboat[0], lboat[1], alpha=0.5)
	plt.plot(x, y, "ok")



def lcoastfun(y):
	[Ycoast, L, dL, R, dR] = smooth_coast.T
	return interpol(y, Ycoast, L, dL)

def rcoastfun(y):
	[Ycoast, L, dL, R, dR] = smooth_coast.T
	return interpol(y, Ycoast, R, dR)


y = np.linspace(-2000, 2000, 301)



px, py = posfun(t)

lt = 850
plotboat(*posfun(lt), hdgfun(lt))
plt.plot(px, py)
plt.ylim([1500, 2000])
plt.xlim([-250, 250])
plt.gca().set_aspect(1)
plt.show()


tees = np.linspace(600, 650, 10)
xmax = []
for i, lt in enumerate(tees):
	fig = plt.figure(figsize=defsize*[2,3])
	plt.plot(lcoastfun(y), y, "k")
	plt.plot(rcoastfun(y), y, "k")

	plotboat(*posfun(lt), hdgfun(lt))
	plt.plot(px, py)
	plt.ylim([0, 2000])
	plt.xlim([-250, 250])
	plt.gca().set_aspect(1)
	plt.show()

# %%
tees = np.linspace(650, 850, 10)
xmax = []
for i, lt in enumerate(tees):
	fig = plt.figure(figsize=defsize*[2,3])
	plt.plot(lcoastfun(y), y, "k")
	plt.plot(rcoastfun(y), y, "k")

	plotboat(*posfun(lt), hdgfun(lt))
	plt.plot(px, py)
	plt.ylim([0, 2000])
	plt.xlim([-250, 250])
	plt.gca().set_aspect(1)
	plt.show()


# %%

# %%

# %%
