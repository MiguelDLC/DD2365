#%% 
import numpy as np
from matplotlib import pyplot as plt
from time import perf_counter as cl
from utils import *

# %%
import matplotlib
matplotlib.rc("text", usetex=True)
matplotlib.rc("font", family="serif")

matplotlib.rc("text.latex", preamble=r"""
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{bm}
\DeclareMathOperator{\newdiff}{d} % use \dif instead
\newcommand{\dif}{\newdiff\!} %the correct way to do derivatives
\newcommand{\bigoh}{\mathcal{O}}
\makeatletter
\let\oldabs\abs
\def\abs{\@ifstar{\oldabs}{\oldabs*}}
\newcommand\norm[1]{\left\lVert#1\right\rVert}
""")

c = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
os.makedirs("Report/Figures", exist_ok=True)

#%%

t = np.linspace(0, 10, 1001)
x, y = pos_com(t)
l, r = lcoastfun(y*1.5), rcoastfun(y*1.5)

T = [3, 4.5, 6, 7, 9]
fig, axes = plt.subplots(1, len(T), sharex=True, figsize=defsize*[2, 1])
for (i, t) in enumerate(T):
	ax = axes[i]
	ax.set_title("t = %.2d:%0.2d:%0.2d" % hms(t*L/U))
	ax.plot(x, y)
	ax.plot(l, 1.5*y, "k")
	ax.plot(r, 1.5*y, "k")
	xc, yc = pos_com(t)
	boat = boatfun(t)
	ax.fill(boat[0], boat[1], color="white", alpha=0.5, zorder=10)
	ax.plot(boat[0], boat[1], color="k", lw=1, zorder=20)
	ax.plot(xc, yc, 'ok', lw=0, ms=3, zorder=20)
	ax.set_ylim([yc-0.9, yc+0.9])
	ax.set_aspect(1)
	ax.set_xlabel("$x/L$")
axes[0].set_ylabel("$y/L$")
plt.tight_layout()
plt.savefig("Report/Figures/snaps.pdf")


#%%


t = np.linspace(0, 10, 1001)
x, y = pos_com(t)
l, r = lcoastfun(y*1.5), rcoastfun(y*1.5)

T = [7, 7.6, 8]
fig, axes = plt.subplots(1, len(T), sharex=True, figsize=defsize*[1.5, 1])
for (i, t) in enumerate(T):
	ax = axes[i]

	ax.plot(l, 1.5*y, "k")
	ax.plot(r, 1.5*y, "k")
	xc, yc = pos_com(t)
	boat = boatfun(t)
	xmin, xmax = lcoastfun(t), rcoastfun(t)
	ref=30
	th = 1.5*2/ref
	boat2, _ = updated_boat_geo(boat, xmin, xmax, ref)

	ax.fill(boat[0], boat[1], color="k", alpha=0.15, zorder=-2)
	ax.plot(boat2[0], boat2[1], color="k", lw=0.5, zorder=20)
	ax.plot(xc, yc, 'ok', lw=0, ms=3, zorder=20)
	ax.set_ylim([yc-0.7, yc+0.7])

	mesh, meshtype, move_info, flow_vars, flow_params = remesh(boat, ref)
	plt.sca(ax)

	plt.axvline(xmin+th, ls="--", lw=0.5, c="k")
	plt.axvline(xmax-th, ls="--", lw=0.5, c="k")

	dl.plot(mesh, linewidth=0.5, alpha=0.5, zorder=21)

	ax.set_aspect(1)
	ax.set_xlabel("$x/L$")
axes[0].set_ylabel("$y/L$")
plt.tight_layout()
plt.savefig("Report/Figures/topology.pdf")





# ==========================================================================
# First experiment : compute the forces
#%% 
ref = 50
t0 = 0.0
boat = boatfun(t0)
mesh, meshtype, move_info, flow_vars, flow_params = remesh(boat, ref)
hmax = mesh.hmax()

u0, u1, p1 = flow_vars
au, Lu, ap, Lp, bcu, bcp, hmin, mydt, boatspeedx, boatspeedy, V, forces_form = flow_params
Forcex, Forcey, Torque, Rvec = forces_form
Ad, bd, d, boatmovx, boatmovy, dom_move, bcd = move_info


# Set parameters for nonlinear and lienar solvers 
num_nnlin_iter = 5
prec = "amg" if dl.has_krylov_solver_preconditioner("amg") else "default" 

t = t0
i = 0

#%%

force_array = [[0, 0, 0]]
t_array = [t0]
since_remesh = 0
t_remesh = []
i_remesh = []


remeshfig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=defsize*[0.9*1.5, 1])


while t < 11:
	if since_remesh < 15:
		curr_dt = hmin/15 * 0.5
	else:
		curr_dt = hmin * 0.5
	mydt.dt = curr_dt

	x0, y0 = pos_com(t)
	x1, y1 = pos_com(t+curr_dt)
	dx = x1 - x0
	dy = y1 - y0
	dtheta = hdgfun(t) - hdgfun(t+curr_dt)

	w =  1/(1+np.exp(-15*(since_remesh/20 -1)))

	# Set mesh movement before fluid computation
	set_dom_move(x0, y0, dx, dy, dtheta, boatmovx, boatmovy, dom_move)
	[bc.apply(Ad, bd) for bc in bcd]
	[bc.apply(d.vector()) for bc in bcd]
	dl.solve(Ad, d.vector(), bd, "bicgstab", "default")

	
	# fluid computation
	set_speeds(x0, y0, dx, dy, dtheta, curr_dt, boatspeedx, boatspeedy)
	k = 0
	for k in range(num_nnlin_iter):
		# Assemble momentum matrix and vector 
		Au = dl.assemble(au)
		bu = dl.assemble(Lu)

		# Compute velocity solution 
		[bc.apply(Au, bu) for bc in bcu]
		[bc.apply(u1.vector()) for bc in bcu]
		dl.solve(Au, u1.vector(), bu, "bicgstab", "default")

		# Assemble continuity matrix and vector
		Ap = dl.assemble(ap) 
		bp = dl.assemble(Lp)

		# Compute pressure solution 
		[bc.apply(Ap, bp) for bc in bcp]
		[bc.apply(p1.vector()) for bc in bcp]
		dl.solve(Ap, p1.vector(), bp, "bicgstab", prec)


	u0.assign(u1)
	dl.ALE.move(mesh, d)
	t += curr_dt
	i += 1
	since_remesh += 1

	# Force computation
	xc, yc = pos_com(t)
	Rvec.xc = xc
	Rvec.yc = yc
	fx1 = dl.assemble(Forcex)
	fy1 = dl.assemble(Forcey)
	tor1 = dl.assemble(Torque)
	t_array += [t]
	fx0, fy0, tor0 = force_array[-1]
	fx, fy, tor = fx0*(1-w) + w*fx1, fy0*(1-w) + w*fy1, tor0*(1-w) + w*tor1
	force_array += [[fx, fy, tor]]

	
	xmin, xmax = lcoastfun(yc), rcoastfun(yc)
	ymin, ymax = yc-800/L, yc+800/L

	if i ==298 or i == 299 or i == 315:
		ax = axes[min(i-298, 2)]
		ax.set_title("i = %d" % i)
		plt.sca(ax)
		
		levels = np.linspace(-1.1, 1.1, 23)/2
		pcol = dl.plot(p1, alpha=0.5, cmap="Spectral_r", vmin=-1/2, vmax=1/2, levels=levels, extend="both")
		dl.plot(u1)
		dl.plot(mesh, linewidth=0.5, alpha=0.5)
		plt.plot(*pos_com(t), 'o')
		plt.arrow(xc, yc, fx1/2, fy1/2, length_includes_head=True, head_width=2e-2)

		yc = (ymin + ymax)/2
		plt.xlim([xmin, xmax])
		plt.ylim([yc-0.7, yc+0.7])
		ax.set_aspect(1)
		ax.set_xlabel("$x/L$")
		axes[0].set_ylabel("$y/L$")


	if i%10 == 0:
		print("i = %d, t=%.3f" % (i,t))

	boat = boatfun(t)
	_, new_meshtype = updated_boat_geo(boat, xmin, xmax, ref)

	if MeshQuality.radius_ratio_min_max(mesh)[0] < 0.2 or mesh.hmax() > 1.5*hmax or new_meshtype != meshtype:
		u1_old = u1

		mesh, meshtype, move_info, flow_vars, flow_params = remesh(boat, ref)
		hmax = mesh.hmax()

		u0, u1, p1 = flow_vars
		au, Lu, ap, Lp, bcu, bcp, hmin, mydt, boatspeedx, boatspeedy, V, forces_form = flow_params
		Forcex, Forcey, Torque, Rvec = forces_form
		Ad, bd, d, boatmovx, boatmovy, dom_move, bcd = move_info
		print("Remesh at iter %d, t=%.5g, New hmin=%.3g" % (i, t, hmin))

		unew = dl.interpolate(u1_old, V)
		u1.assign(unew)
		u0.assign(unew)
		[bc.apply(u1.vector()) for bc in bcu]
		since_remesh = 0
		t_remesh += [t]
		i_remesh += [i]

plt.tight_layout()
plt.savefig("Report/Figures/remesh.pdf")
plt.show()

# %% =========================

force_array = np.array(force_array).T
t_array = np.array(t_array)
forcex_array = force_array[0]
forcey_array = force_array[1]
torque_array = force_array[2]
i_remesh = np.array(i_remesh)

#%%
fig = plt.figure(figsize=defsize*[4,2])

mt = np.copy(t_array)
Tor = torque_array
Fx = np.copy(forcex_array)
Fy = np.copy(forcey_array)
Fxy = np.array([Fx, Fy])
Fsf = np.einsum("nij,jn->in", rotate_matrix(hdgfun(mt)), Fxy)
Fside = Fsf[0]
Ffront = Fsf[1]


plt.plot(mt, Fside, c=c[0], label="Lateral force")
plt.plot(mt, Ffront, c=c[1], label="Longitudinal force")
plt.plot(mt, Tor, c=c[3], label="Torque")

# for tc in ((smoothpath.T)[0]-TINIT)*U/L:
# 	plt.axvline(tc, ls="--", linewidth=0.5, color=c[2], zorder=10)

for tc in t_remesh:
	plt.axvline(tc, ls=":", linewidth=0.5, color="k", zorder=10)

#plt.plot(mt, w/2)
plt.xlim([0, 10])
plt.ylim([-1, 1])

for y_ in np.linspace(-1, 1, 9):
	plt.axhline(y_, linewidth=0.5, alpha=0.5, zorder=-2000)


axfun = lambda t : (pos_com(t - 1e-4)[0] - 2*pos_com(t)[0] + pos_com(t + 1e-4)[0])/1e-8
ayfun = lambda t : (pos_com(t - 1e-4)[1] - 2*pos_com(t)[1] + pos_com(t + 1e-4)[1])/1e-8
domegafun = lambda t : - (hdgfun(t - 1e-4) - 2*hdgfun(t) + hdgfun(t + 1e-4))/1e-8
fxfun = interp1d(mt, Fx)
fyfun = interp1d(mt, Fy)
couplefun = interp1d(mt, Tor)

avgthrust = -np.mean(Ffront[mt < 5])
enginefun = lambda hdg : rotate_matrix(-hdg).dot([0, avgthrust]).T

externalfxfun = lambda t : BoatMass/M_0*axfun(t) - fxfun(t) - enginefun(hdgfun(t))[0]
externalfyfun = lambda t : BoatMass/M_0*ayfun(t) - fyfun(t) - enginefun(hdgfun(t))[1]
Inertia = compute_inertia()
externalcouplefun = lambda t : Inertia * BoatMass/M_0*domegafun(t) - couplefun(t)


axy = np.array([axfun(mt), ayfun(mt)])
asf = np.einsum("nij,jn->in", rotate_matrix(hdgfun(mt)), axy)
aside = asf[0]
afront = asf[1]


plt.plot(mt, BoatMass/M_0 * aside, "--", lw=0.7, c=c[0], label=r"$m/M_0 \cdot \hat{a}_\text{lateral}$")
plt.plot(mt, BoatMass/M_0 * afront, "--", lw=0.7, c=c[1], label=r"$m/M_0 \cdot \hat{a}_\text{longitudinal}$")
plt.plot(mt, BoatMass/M_0 * Inertia * domegafun(mt), "--", lw=0.7, c=c[3], label=r"$\hat{I} m/M_0 \cdot \dot{\hat{\omega}}$")
plt.legend(fontsize=12)
plt.xlabel(r"$\hat{t}$")
plt.tight_layout()
plt.savefig("Report/Figures/forces.pdf")

#%%
fig = plt.figure(figsize=defsize*[4,2])

plt.plot(mt, Fside, c=c[0], label="Lateral force")
plt.plot(mt, Ffront, c=c[1], label="Longitudinal force")
plt.plot(mt, Tor, c=c[3], label="Torque")

for tc in t_remesh:
	plt.axvline(tc, ls=":", linewidth=0.5, color="k", zorder=10)

for y_ in np.linspace(-1, 1, 9):
	plt.axhline(y_, linewidth=0.5, alpha=0.5, zorder=-2000)


plt.plot(mt, 15*BoatMass/M_0 * aside, "--", lw=0.7, c=c[0], label=r"$15m/M_0 \cdot \hat{a}_\text{lateral}$")
plt.plot(mt, 15*BoatMass/M_0 * afront, "--", lw=0.7, c=c[1], label=r"$15m/M_0 \cdot \hat{a}_\text{longitudinal}$")
plt.plot(mt, 15*BoatMass/M_0 * Inertia * domegafun(mt), "--", lw=0.7, c=c[3], label=r"$15\hat{I} m/M_0 \cdot \dot{\hat{\omega}}$")
plt.xlim([0, 10])
plt.ylim([-1, 1])

plt.legend(fontsize=12)
plt.xlabel(r"$\hat{t}$")
plt.tight_layout()
plt.savefig("Report/Figures/forces_wrong.pdf")





# ==========================================================================
# Second experiment : recreate the event from the forces
# %%

ref = 50
t0 = 0.0
boat = boatfun(t0)
mesh, meshtype, move_info, flow_vars, flow_params = remesh(boat, ref)
hmax = mesh.hmax()

u0, u1, p1 = flow_vars
au, Lu, ap, Lp, bcu, bcp, hmin, mydt, boatspeedx, boatspeedy, V, forces_form = flow_params
Forcex, Forcey, Torque, Rvec = forces_form
Ad, bd, d, boatmovx, boatmovy, dom_move, bcd = move_info

# Set parameters for nonlinear and lienar solvers 
num_nnlin_iter = 5
prec = "amg" if dl.has_krylov_solver_preconditioner("amg") else "default" 

t = t0
i = 0

# ==============================
#%%
force_array = [[0, 0, 0]]
t_array = [t0]
since_remesh = 0
t_remesh = []
i_remesh = []

# pre-starting the sim from historical data
while t < 0.5:
	if since_remesh < 15:
		curr_dt = hmin/15 * 0.5
	else:
		curr_dt = hmin * 0.5
	mydt.dt = curr_dt

	x0, y0 = pos_com(t)
	x1, y1 = pos_com(t+curr_dt)
	dx = x1 - x0
	dy = y1 - y0
	dtheta = hdgfun(t) - hdgfun(t+curr_dt)

	w =  1/(1+np.exp(-15*(since_remesh/20 -1)))

	# Set mesh movement before fluid computation
	set_dom_move(x0, y0, dx, dy, dtheta, boatmovx, boatmovy, dom_move)
	[bc.apply(Ad, bd) for bc in bcd]
	[bc.apply(d.vector()) for bc in bcd]
	dl.solve(Ad, d.vector(), bd, "bicgstab", "default")

	
	# fluid computation
	set_speeds(x0, y0, dx, dy, dtheta, curr_dt, boatspeedx, boatspeedy)
	k = 0
	for k in range(num_nnlin_iter):
		# Assemble momentum matrix and vector 
		Au = dl.assemble(au)
		bu = dl.assemble(Lu)

		# Compute velocity solution 
		[bc.apply(Au, bu) for bc in bcu]
		[bc.apply(u1.vector()) for bc in bcu]
		dl.solve(Au, u1.vector(), bu, "bicgstab", "default")

		# Assemble continuity matrix and vector
		Ap = dl.assemble(ap) 
		bp = dl.assemble(Lp)

		# Compute pressure solution 
		[bc.apply(Ap, bp) for bc in bcp]
		[bc.apply(p1.vector()) for bc in bcp]
		dl.solve(Ap, p1.vector(), bp, "bicgstab", prec)


	u0.assign(u1)
	dl.ALE.move(mesh, d)
	t += curr_dt
	i += 1
	since_remesh += 1

	# Force computation
	xc, yc = pos_com(t)
	Rvec.xc = xc
	Rvec.yc = yc
	fx1 = dl.assemble(Forcex)
	fy1 = dl.assemble(Forcey)
	tor1 = dl.assemble(Torque)
	t_array += [t]
	fx0, fy0, tor0 = force_array[-1]
	fx, fy, tor = fx0*(1-w) + w*fx1, fy0*(1-w) + w*fy1, tor0*(1-w) + w*tor1
	force_array += [[fx, fy, tor]]


	if i%10 == 0:
		print("i = %d, t=%.3f" % (i,t))

	xmin, xmax = lcoastfun(yc), rcoastfun(yc)
	boat = boatfun(t)
	_, new_meshtype = updated_boat_geo(boat, xmin, xmax, ref)

	if MeshQuality.radius_ratio_min_max(mesh)[0] < 0.2 or mesh.hmax() > 1.5*hmax or new_meshtype != meshtype:
		u1_old = u1

		mesh, meshtype, move_info, flow_vars, flow_params = remesh(boat, ref)
		hmax = mesh.hmax()

		u0, u1, p1 = flow_vars
		au, Lu, ap, Lp, bcu, bcp, hmin, mydt, boatspeedx, boatspeedy, V, forces_form = flow_params
		Forcex, Forcey, Torque, Rvec = forces_form
		Ad, bd, d, boatmovx, boatmovy, dom_move, bcd = move_info
		print("Remesh at iter %d, t=%.5g, New hmin=%.3g" % (i, t, hmin))

		unew = dl.interpolate(u1_old, V)
		u1.assign(unew)
		u0.assign(unew)
		[bc.apply(u1.vector()) for bc in bcu]
		since_remesh = 0
		t_remesh += [t]
		i_remesh += [i]


# %% ======= and now the simulation

x0, y0 = pos_com(t-0.5e-5)
x1, y1 = pos_com(t+0.5e-5)
hdg0 = hdgfun(t-0.5e-5)
hdg1 = hdgfun(t+0.5e-5)
hdg = hdgfun(t)

vx = (x1-x0)/1e-5
vy = (y1-y0)/1e-5
omega = (hdg0-hdg1)/1e-5

Inertia = compute_inertia()

xc, yc = pos_com(t) #position of center of mass

Fx0, Fy0, Tau0 = force_array[-1]
Fx1, Fy1, Tau1 = Fx0, Fy0, Tau0


#%%

numplot = 0
Tplots = [0.55, 2, 4, 5, 6, 7]
snapfig, axes = plt.subplots(1, len(Tplots), sharex=True, figsize=defsize*[2, 1])

while t < 9 and i < 635:
	if since_remesh < 15:
		curr_dt = hmin/15 * 0.5
	else:
		curr_dt = hmin * 0.5
	mydt.dt = curr_dt
	w =  1/(1+np.exp(-15*(since_remesh/20 -1))) * 0.5


	
	# fluid computation
	k = 0
	for k in range(num_nnlin_iter*2):
		dvx = M_0/BoatMass * (Fx0 + Fx1 + externalfxfun(t) + externalfxfun(t+curr_dt) + enginefun(hdgfun(t))[0] + enginefun(hdgfun(t+curr_dt))[0])/2 * curr_dt
		dvy = M_0/BoatMass * (Fy0 + Fy1 + externalfyfun(t) + externalfyfun(t+curr_dt) + enginefun(hdgfun(t))[1] + enginefun(hdgfun(t+curr_dt))[1])/2 * curr_dt
		domega = M_0/(BoatMass * Inertia) * (Tau0 + Tau1 + externalcouplefun(t) + externalcouplefun(t+curr_dt))/2 * curr_dt

		dx = (vx + vx+ dvx)/2 * curr_dt
		dy = (vy + vy+ dvy)/2 * curr_dt
		dtheta = (omega + omega+2*domega)/2 * curr_dt

		# Set mesh movement before fluid computation
		set_dom_move(xc, yc, dx, dy, dtheta, boatmovx, boatmovy, dom_move)
		[bc.apply(Ad, bd) for bc in bcd]
		[bc.apply(d.vector()) for bc in bcd]
		dl.solve(Ad, d.vector(), bd, "bicgstab", "default")

		set_speeds(xc, yc, dx, dy, dtheta, curr_dt, boatspeedx, boatspeedy)
		# Assemble momentum matrix and vector 
		Au = dl.assemble(au)
		bu = dl.assemble(Lu)

		# Compute velocity solution 
		[bc.apply(Au, bu) for bc in bcu]
		[bc.apply(u1.vector()) for bc in bcu]
		dl.solve(Au, u1.vector(), bu, "bicgstab", "default")

		# Assemble continuity matrix and vector
		Ap = dl.assemble(ap) 
		bp = dl.assemble(Lp)

		# Compute pressure solution 
		[bc.apply(Ap, bp) for bc in bcp]
		[bc.apply(p1.vector()) for bc in bcp]
		dl.solve(Ap, p1.vector(), bp, "bicgstab", prec)

		# Force computation
		
		Rvec.xc = xc
		Rvec.yc = yc

		dl.ALE.move(mesh, d)
		fx1 = dl.assemble(Forcex)
		fy1 = dl.assemble(Forcey)
		tor1 = dl.assemble(Torque)

		set_dom_move(xc+dx, yc+dy, -dx, -dy, -dtheta, boatmovx, boatmovy, dom_move)
		[bc.apply(Ad, bd) for bc in bcd]
		[bc.apply(d.vector()) for bc in bcd]
		dl.solve(Ad, d.vector(), bd, "bicgstab", "default")
		dl.ALE.move(mesh, d)

		Fx1, Fy1, Tau1 = Fx0*(1-w) + w*fx1, Fy0*(1-w) + w*fy1, Tau0*(1-w) + w*tor1

	set_dom_move(xc, yc, dx, dy, dtheta, boatmovx, boatmovy, dom_move)
	[bc.apply(Ad, bd) for bc in bcd]
	[bc.apply(d.vector()) for bc in bcd]
	dl.solve(Ad, d.vector(), bd, "bicgstab", "default")

	omega += domega
	vx += dvx
	vy += dvy
	xc += dx
	yc += dy
	hdg -= dtheta

	u0.assign(u1)
	dl.ALE.move(mesh, d)
	t += curr_dt
	i += 1
	since_remesh += 1

	t_array += [t]
	force_array += [[Fx1, Fy1, Tau1]]
	Fx0, Fy0, Tau0 = Fx1, Fy1, Tau1
	fx, fy = Fx1, Fy1


	if i%10 == 0:
		print("i = %d, t=%.3f" % (i,t))
	

	xmin, xmax = lcoastfun(yc), rcoastfun(yc)
	if numplot < len(Tplots) and t > Tplots[numplot]:
		ax = axes[numplot]
		ax.set_title(r"$\hat{t} = %0.3f$" % t)
		plt.sca(ax)

		levels = np.linspace(-1.1, 1.1, 23)/2
		pcol = dl.plot(p1, alpha=0.5, cmap="Spectral_r", vmin=-1/2, vmax=1/2, levels=levels, extend="both")
		dl.plot(u1)
		dl.plot(mesh, linewidth=0.5, alpha=0.5)
		history_boat = boatfun(t)
		plt.plot(history_boat[0], history_boat[1],"k", lw=0.5)
		plt.plot(xc, yc, 'o')
		plt.arrow(xc, yc, fx/2, fy/2, length_includes_head=True, head_width=2e-2)

		ax.set_ylim([yc-0.9, yc+0.9])
		ymin, ymax = [yc-0.9, yc+0.9]

		plt.xlim([xmin, xmax])
		
		ax.set_aspect(1)
		ax.set_xlabel(r"$\hat{x}$")
		numplot += 1 


	boat = boatfun2(xc, yc, hdg)
	_, new_meshtype = updated_boat_geo(boat, xmin, xmax, ref)

	if MeshQuality.radius_ratio_min_max(mesh)[0] < 0.2 or mesh.hmax() > 1.5*hmax or new_meshtype != meshtype:
		u1_old = u1

		mesh, meshtype, move_info, flow_vars, flow_params = remesh(boat, ref)
		hmax = mesh.hmax()

		u0, u1, p1 = flow_vars
		au, Lu, ap, Lp, bcu, bcp, hmin, mydt, boatspeedx, boatspeedy, V, forces_form = flow_params
		Forcex, Forcey, Torque, Rvec = forces_form
		Ad, bd, d, boatmovx, boatmovy, dom_move, bcd = move_info
		print("Remesh at iter %d, t=%.5g, New hmin=%.3g" % (i, t, hmin))

		unew = dl.interpolate(u1_old, V)
		u1.assign(unew)
		u0.assign(unew)
		[bc.apply(u1.vector()) for bc in bcu]
		since_remesh = 0
		t_remesh += [t]
		i_remesh += [i]

axes[0].set_ylabel(r"$\hat{y}$")
plt.sca(axes[0])
plt.tight_layout()
plt.savefig("Report/Figures/simu_snaps.pdf")


# %%


# ==========================================================================
# Third experiment : correct reality



ref = 50
t0 = 0.0
boat = boatfun(t0)
mesh, meshtype, move_info, flow_vars, flow_params = remesh(boat, ref)
hmax = mesh.hmax()

u0, u1, p1 = flow_vars
au, Lu, ap, Lp, bcu, bcp, hmin, mydt, boatspeedx, boatspeedy, V, forces_form = flow_params
Forcex, Forcey, Torque, Rvec = forces_form
Ad, bd, d, boatmovx, boatmovy, dom_move, bcd = move_info

# Set parameters for nonlinear and lienar solvers 
num_nnlin_iter = 5
prec = "amg" if dl.has_krylov_solver_preconditioner("amg") else "default" 

t = t0
i = 0

# ==============================

force_array = [[0, 0, 0]]
t_array = [t0]
since_remesh = 0
t_remesh = []
i_remesh = []

# pre-starting the sim from historical data
while t < 0.5:
	if since_remesh < 15:
		curr_dt = hmin/15 * 0.5
	else:
		curr_dt = hmin * 0.5
	mydt.dt = curr_dt

	x0, y0 = pos_com(t)
	x1, y1 = pos_com(t+curr_dt)
	dx = x1 - x0
	dy = y1 - y0
	dtheta = hdgfun(t) - hdgfun(t+curr_dt)

	w =  1/(1+np.exp(-15*(since_remesh/20 -1)))

	# Set mesh movement before fluid computation
	set_dom_move(x0, y0, dx, dy, dtheta, boatmovx, boatmovy, dom_move)
	[bc.apply(Ad, bd) for bc in bcd]
	[bc.apply(d.vector()) for bc in bcd]
	dl.solve(Ad, d.vector(), bd, "bicgstab", "default")

	
	# fluid computation
	set_speeds(x0, y0, dx, dy, dtheta, curr_dt, boatspeedx, boatspeedy)
	k = 0
	for k in range(num_nnlin_iter):
		# Assemble momentum matrix and vector 
		Au = dl.assemble(au)
		bu = dl.assemble(Lu)

		# Compute velocity solution 
		[bc.apply(Au, bu) for bc in bcu]
		[bc.apply(u1.vector()) for bc in bcu]
		dl.solve(Au, u1.vector(), bu, "bicgstab", "default")

		# Assemble continuity matrix and vector
		Ap = dl.assemble(ap) 
		bp = dl.assemble(Lp)

		# Compute pressure solution 
		[bc.apply(Ap, bp) for bc in bcp]
		[bc.apply(p1.vector()) for bc in bcp]
		dl.solve(Ap, p1.vector(), bp, "bicgstab", prec)


	u0.assign(u1)
	dl.ALE.move(mesh, d)
	t += curr_dt
	i += 1
	since_remesh += 1

	# Force computation
	xc, yc = pos_com(t)
	Rvec.xc = xc
	Rvec.yc = yc
	fx1 = dl.assemble(Forcex)
	fy1 = dl.assemble(Forcey)
	tor1 = dl.assemble(Torque)
	t_array += [t]
	fx0, fy0, tor0 = force_array[-1]
	fx, fy, tor = fx0*(1-w) + w*fx1, fy0*(1-w) + w*fy1, tor0*(1-w) + w*tor1
	force_array += [[fx, fy, tor]]


	if i%10 == 0:
		print("i = %d, t=%.3f" % (i,t))

	xmin, xmax = lcoastfun(yc), rcoastfun(yc)
	boat = boatfun(t)
	_, new_meshtype = updated_boat_geo(boat, xmin, xmax, ref)

	if MeshQuality.radius_ratio_min_max(mesh)[0] < 0.2 or mesh.hmax() > 1.5*hmax or new_meshtype != meshtype:
		u1_old = u1

		mesh, meshtype, move_info, flow_vars, flow_params = remesh(boat, ref)
		hmax = mesh.hmax()

		u0, u1, p1 = flow_vars
		au, Lu, ap, Lp, bcu, bcp, hmin, mydt, boatspeedx, boatspeedy, V, forces_form = flow_params
		Forcex, Forcey, Torque, Rvec = forces_form
		Ad, bd, d, boatmovx, boatmovy, dom_move, bcd = move_info
		print("Remesh at iter %d, t=%.5g, New hmin=%.3g" % (i, t, hmin))

		unew = dl.interpolate(u1_old, V)
		u1.assign(unew)
		u0.assign(unew)
		[bc.apply(u1.vector()) for bc in bcu]
		since_remesh = 0
		t_remesh += [t]
		i_remesh += [i]



# %%

x0, y0 = pos_com(t-0.5e-5)
x1, y1 = pos_com(t+0.5e-5)
hdg0 = hdgfun(t-0.5e-5)
hdg1 = hdgfun(t+0.5e-5)
hdg = hdgfun(t)

vx = (x1-x0)/1e-5
vy = (y1-y0)/1e-5
omega = (hdg0-hdg1)/1e-5

Inertia = compute_inertia()

xc, yc = pos_com(t) #position of center of mass

Fx0, Fy0, Tau0 = force_array[-1]
Fx1, Fy1, Tau1 = Fx0, Fy0, Tau0

#%%

numplot = 0
Tplots = [2, 4, 6.5, 7.3, 8]
snapfig, axes = plt.subplots(1, len(Tplots), sharex=True, figsize=defsize*[2, 1])


def control(t):
	if t<1.5:
		return 0
	elif t < 4.2:
		return -0.01
	else:
		return 0.02

while t < 9 and i < 700:
	if since_remesh < 15:
		curr_dt = hmin/15 * 0.5
	else:
		curr_dt = hmin * 0.5
	mydt.dt = curr_dt
	w =  1/(1+np.exp(-15*(since_remesh/20 -1))) * 0.2


	dtheta = 0.0
	# fluid computation
	k = 0
	for k in range(num_nnlin_iter*2):
		dvx = M_0/BoatMass * (Fx0 + Fx1 + externalfxfun(t) + externalfxfun(t+curr_dt) + enginefun(hdg)[0] + enginefun(hdg - dtheta)[0])/2 * curr_dt
		dvy = M_0/BoatMass * (Fy0 + Fy1 + externalfyfun(t) + externalfyfun(t+curr_dt) + enginefun(hdg)[1] + enginefun(hdg - dtheta)[1])/2 * curr_dt
		domega = M_0/(BoatMass * Inertia) * (Tau0 + Tau1 + externalcouplefun(t) + externalcouplefun(t+curr_dt) + 2*control(t))/2 * curr_dt

		dx = (vx + vx+ dvx)/2 * curr_dt
		dy = (vy + vy+ dvy)/2 * curr_dt
		dtheta = (omega + omega+2*domega)/2 * curr_dt


		# Set mesh movement before fluid computation
		set_dom_move(xc, yc, dx, dy, dtheta, boatmovx, boatmovy, dom_move)
		[bc.apply(Ad, bd) for bc in bcd]
		[bc.apply(d.vector()) for bc in bcd]
		dl.solve(Ad, d.vector(), bd, "bicgstab", "default")



		set_speeds(xc, yc, dx, dy, dtheta, curr_dt, boatspeedx, boatspeedy)
		# Assemble momentum matrix and vector 
		Au = dl.assemble(au)
		bu = dl.assemble(Lu)

		# Compute velocity solution 
		[bc.apply(Au, bu) for bc in bcu]
		[bc.apply(u1.vector()) for bc in bcu]
		dl.solve(Au, u1.vector(), bu, "bicgstab", "default")

		# Assemble continuity matrix and vector
		Ap = dl.assemble(ap) 
		bp = dl.assemble(Lp)

		# Compute pressure solution 
		[bc.apply(Ap, bp) for bc in bcp]
		[bc.apply(p1.vector()) for bc in bcp]
		dl.solve(Ap, p1.vector(), bp, "bicgstab", prec)

		# Force computation
		
		Rvec.xc = xc
		Rvec.yc = yc

		dl.ALE.move(mesh, d)
		fx1 = dl.assemble(Forcex)
		fy1 = dl.assemble(Forcey)
		tor1 = dl.assemble(Torque)

		set_dom_move(xc+dx, yc+dy, -dx, -dy, -dtheta, boatmovx, boatmovy, dom_move)
		[bc.apply(Ad, bd) for bc in bcd]
		[bc.apply(d.vector()) for bc in bcd]
		dl.solve(Ad, d.vector(), bd, "bicgstab", "default")
		dl.ALE.move(mesh, d)



		Fx1, Fy1, Tau1 = Fx0*(1-w) + w*fx1, Fy0*(1-w) + w*fy1, Tau0*(1-w) + w*tor1

	set_dom_move(xc, yc, dx, dy, dtheta, boatmovx, boatmovy, dom_move)
	[bc.apply(Ad, bd) for bc in bcd]
	[bc.apply(d.vector()) for bc in bcd]
	dl.solve(Ad, d.vector(), bd, "bicgstab", "default")

	omega += domega
	vx += dvx
	vy += dvy
	xc += dx
	yc += dy
	hdg -= dtheta

	u0.assign(u1)
	dl.ALE.move(mesh, d)
	t += curr_dt
	i += 1
	since_remesh += 1

	t_array += [t]
	force_array += [[Fx1, Fy1, Tau1]]
	Fx0, Fy0, Tau0 = Fx1, Fy1, Tau1
	fx, fy = Fx1, Fy1

	if i%10 == 0:
		print("i = %d, t=%.3f" % (i,t))

	
	xmin, xmax = lcoastfun(yc), rcoastfun(yc)
	if numplot < len(Tplots) and t > Tplots[numplot]:
		ax = axes[numplot]
		ax.set_title(r"$\hat{t} = %0.3f$" % t)
		plt.sca(ax)

		levels = np.linspace(-1.1, 1.1, 23)/2
		pcol = dl.plot(p1, alpha=0.5, cmap="Spectral_r", vmin=-1/2, vmax=1/2, levels=levels, extend="both")
		dl.plot(u1)
		dl.plot(mesh, linewidth=0.5, alpha=0.5)
		history_boat = boatfun(t)
		plt.plot(history_boat[0], history_boat[1],"k", lw=0.5)
		plt.plot(xc, yc, 'o')
		plt.arrow(xc, yc, fx/2, fy/2, length_includes_head=True, head_width=2e-2)

		ax.set_ylim([yc-0.9, yc+0.9])
		ymin, ymax = [yc-0.9, yc+0.9]

		plt.xlim([xmin, xmax])
		
		ax.set_aspect(1)
		ax.set_xlabel(r"$\hat{x}$")
		numplot += 1 


	boat = boatfun2(xc, yc, hdg)
	_, new_meshtype = updated_boat_geo(boat, xmin, xmax, ref)

	if MeshQuality.radius_ratio_min_max(mesh)[0] < 0.2 or mesh.hmax() > 1.5*hmax or new_meshtype != meshtype:
		u1_old = u1

		mesh, meshtype, move_info, flow_vars, flow_params = remesh(boat, ref)
		hmax = mesh.hmax()

		u0, u1, p1 = flow_vars
		au, Lu, ap, Lp, bcu, bcp, hmin, mydt, boatspeedx, boatspeedy, V, forces_form = flow_params
		Forcex, Forcey, Torque, Rvec = forces_form
		Ad, bd, d, boatmovx, boatmovy, dom_move, bcd = move_info
		print("Remesh at iter %d, t=%.5g, New hmin=%.3g" % (i, t, hmin))

		unew = dl.interpolate(u1_old, V)
		u1.assign(unew)
		u0.assign(unew)
		[bc.apply(u1.vector()) for bc in bcu]
		since_remesh = 0
		t_remesh += [t]
		i_remesh += [i]

axes[0].set_ylabel(r"$\hat{y}$")
plt.sca(axes[0])
plt.tight_layout()
plt.savefig("Report/Figures/simu_correct_snaps.pdf")

# %%
