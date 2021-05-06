
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
# %%
