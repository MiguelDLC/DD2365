
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
