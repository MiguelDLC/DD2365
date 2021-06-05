# ------------------------------------------------------------------------------
#
#  Gmsh Python tutorial 1
#
#  Geometry basics, elementary entities, physical groups
#
# ------------------------------------------------------------------------------
#%%
# The Python API is entirely defined in the `gmsh.py' module (which contains the
# full documentation of all the functions in the API):
import gmsh
import sys
import os
import numpy as np
import dolfin as dl
from matplotlib import pyplot as plt 
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

def gmshGenMesh(boat, xmin, xmax, ymin, ymax, ref):
	gmsh.initialize()
	lc = 2/ref #1e-2
	th = 1.5*lc

	N = boat.shape[1]
	xb = boat[0]
	yb = boat[1]

	flagged = xb[:-1] > xmax - th
	pos = np.nonzero(flagged)[0]
	if len(pos) == 1:
		if xb[pos] < xmax:
			pos = pos[0]
			_, [y0, ny1, ny2, y2] = find_nex_pts(xb[:-1][[pos-1, pos, (pos+1)%N]], yb[:-1][[pos-1, pos, (pos+1)%N]], xmax-th)
			
			xb[pos] = xmax-th
			yb[pos] = ny1
			xb = np.insert(xb, pos+1, xmax-th)
			yb = np.insert(yb, pos+1, ny2)

	
	elif len(pos) == 2:
		pos = pos[0]
		px, py , xmed = find_intersect(boat, pos)
		xnew = xmax if xmed > (xmax - th/2) else xmax-th

		_, [y0, ny1, ny2, y2] = find_nex_pts(px, py, xnew)
		xb[pos:pos+2] = xnew
		yb[pos:pos+2] = [ny1, ny2]
	elif len(pos) > 2:
		print("THERE WAS AN ERROR IN THE MESH GENERATION")
		exit(-1)
	boat = np.vstack([xb, yb])
	boat[:,-1] = boat[:,0]


	flagged = xb[:-1] < xmin + th
	pos = np.nonzero(flagged)[0]
	if len(pos) == 1:
		if xb[pos] > xmin:
			pos = pos[0]
			_, [y0, ny1, ny2, y2] = find_nex_pts(xb[:-1][[pos-1, pos, (pos+1)%N]], yb[:-1][[pos-1, pos, (pos+1)%N]], xmin+th)
			
			xb[pos] = xmin+th
			yb[pos] = ny1
			xb = np.insert(xb, pos+1, xmin+th)
			yb = np.insert(yb, pos+1, ny2)

	elif len(pos) == 2:
		pos = pos[0]
		px, py , xmed = find_intersect(boat, pos)
		xnew = xmin if xmed < (xmin + th/2) else xmin+th

		_, [y0, ny1, ny2, y2] = find_nex_pts(px, py, xnew)
		xb[pos:pos+2] = xnew
		yb[pos:pos+2] = [ny1, ny2]
	elif len(pos) > 2:
		print("THERE WAS AN ERROR IN THE MESH GENERATION")
		exit(-1)
	boat = np.vstack([xb, yb])
	boat[:,-1] = boat[:,0]

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
	return mesh




# %%
xmin = -0.3675
xmax = 0.3575
ymin = -2.236457315706494
ymax = 1.0135426842935062
refboat = np.array([
	[-0.07375,  0.07375,  0.07375,  0.     , -0.07375, -0.07375],
	[-0.7    , -0.7    ,  0.15   ,  0.3    ,  0.15   , -0.7    ]])

xmin, xmax, ymin, ymax = -0.367500000000000, 0.357500000000000, 2.293841518901412, 6.293841518901412
refboat = np.array([[-0.37641579, -0.26827815,  0.3131915 ,  0.3611349 ,  0.20505386,
        -0.37641579],
       [ 4.0035956 ,  3.90328359,  4.53011534,  4.69024183,  4.63042736,
         4.0035956 ]])



defsize = np.array([6.4, 4.8])
fig = plt.figure(figsize=defsize*[2,3])
mesh = gmshGenMesh(refboat, xmin, xmax, ymin, ymax, 40)
dl.plot(mesh, linewidth=0.5, alpha=0.5)

# %%

# %%
