
#%%
import numpy as np
from matplotlib import pyplot as plt

import xml.etree.ElementTree as ET
tree = ET.parse('map.xml')
root = tree.getroot()


coastlines = []
nodes = {}

for child in root:
    if child.tag == "node":
        iid = int(child.attrib["id"])
        lat = float(child.attrib["lat"])
        lon = float(child.attrib["lon"])
        coord = [lat, lon]
        nodes[iid] = coord




    if child.tag == "way":
        iid = child.attrib["id"]
        if int(iid) == 387050173 or int(iid) == 387050169:
            coastelems = []
            print(child.attrib)
            for elem in child:
                if elem.tag == "nd":
                    coastelems += [int(elem.attrib["ref"])]
            coastlines += [np.array(coastelems)]

coasts = []
for co in coastlines:
    coords = [nodes[iid] for iid in co]
    coords = np.array(coords)
    coords = coords[coords[:, 0] >= 29.96, :]
    coords = coords[coords[:, 0] <= 30.04, :]
    coasts += [np.array(coords)]


# %%


c0 = np.array(coasts[0])
c1 = np.array(coasts[1])
defsize = np.array([6.4, 4.8])
fig = plt.figure(figsize=2*defsize)

plt.plot(c0[:, 1], c0[:, 0])
plt.plot(c1[:, 1], c1[:, 0])

cX = np.loadtxt("cX.txt", delimiter=",")
cY = np.loadtxt("cY.txt", delimiter=",")
XY = np.loadtxt("XY.txt", delimiter=",")

dx = cX[:, 1].mean()
dy = cY[:, 1].mean()
cx = cX[:, 0] / dx
cy = cY[:, 0] / dy
x = XY[:, 0]  / dx
y = XY[:, 1]  / dy

lat = 29 + 58.5/60 + (-y+cy)*0.3/60
long = 32 + 35.4/60 + (x-cx)*0.3/60


plt.ylim()
plt.plot(long, lat)


plt.gca().set_aspect((np.cos(np.deg2rad(lat)).mean()))
plt.show()

# %%
