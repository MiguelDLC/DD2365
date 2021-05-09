
#%%
import parse
import codecs
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
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
plt.tight_layout()
plt.savefig("../path.png")

# %%




plt.plot(path)
# %%

<<<<<<< HEAD

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




=======
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

[sog, hdg, cog, rot, draught] = [data[:, 1], data[:, 2], data[:, 3], data[:,4], data[:,5]]

cog = np.deg2rad((cog+250) % 360 - 250)
hdg = np.deg2rad((hdg+250) % 360 - 250)

fig = plt.figure(figsize=2*defsize)
plt.plot(t, smooth(sog))
plt.show()

#fig = plt.figure(figsize=2*defsize)
#plt.plot(data[:, 0], data[:, 1])
>>>>>>> 9fc7a8c49a405496f534016aa328c07e5815e213
# %%
