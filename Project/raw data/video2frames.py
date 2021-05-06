#%%
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from numpy.core.numeric import NaN
from scipy.signal import convolve2d
from scipy.optimize import minimize
import pytesseract
import parse
import codecs

#%%

defsize = np.array([6.4, 4.8])

def find_xPos(level, xinit, dxinit, init=False):
    kernel = np.array([[1, -2, 1]])
    lap = convolve2d(level, kernel, mode="same")
    spikes = np.nanmean(lap, axis=0)**2
    spikes[0:5] = 0
    spikes[-6:] = 0
    spikes /= spikes.max()
    index = np.arange(len(spikes))
    if init:
        xinit = np.argmax(spikes)
        dxinit = 116.5

    def fit(X):
        x, dx = X
        pos = (index - x + dx/2) % dx-dx/2
        return np.exp(-pos**2/16)

    def res(X):
        return -spikes.dot(fit(X)) / X[1]

    X0 = [xinit, dxinit]
    x, dx = minimize(res, X0).x

    # plt.plot(index, fit([x, dx]))
    # plt.plot(spikes/spikes.max())
    # print(res([x, dx]))
    # plt.show()
    return x, dx


def find_yPos(level, yinit, dyinit, init=False):
    kernel = np.array([[1, -2, 1]]).T
    lap = convolve2d(level, kernel, mode="same")
    spikes = np.nanmean(lap, axis=1)**2
    spikes[0:5] = 0
    spikes[-6:] = 0
    spikes /= spikes.max()
    index = np.arange(len(spikes))
    if init:
        yinit = np.argmax(spikes)
        dyinit = 134.5

    def fit(X):
        x, dx = X
        pos = (index - x + dx/2) % dx-dx/2
        return np.exp(-pos**2/16)

    def res(X):
        return -spikes.dot(fit(X)) / X[1]

    X0 = [yinit, dyinit]
    y, dy = minimize(res, X0).x

    # plt.plot(index, fit([yinit, dyinit]))
    # plt.plot(spikes/spikes.max())
    # print(res([y, dy]))
    # plt.show()
    return y, dy


def find_boat_pos(level, xinit, yinit, init=False):
    if init:
        xinit, yinit = 488.08, 299.04

    col = np.arange(-10, 11)
    row = np.arange(-10, 11)
    row, col = np.meshgrid(row, col, indexing='ij')

    xbase = int(np.round(xinit))
    ybase = int(np.round(yinit))

    dxinit = xinit - xbase
    dyinit = yinit - ybase

    loclevel = level[ybase + row, xbase + col]
    loclevel -= np.median(loclevel)
    loclevel = np.clip(loclevel, 0, 255)

    def fit(X):
        dx, dy = X
        return np.exp(-((col+dx)**2 + (row + dy)**2)/2)

    def res(X):
        return -np.einsum("ij,ij->", loclevel, fit(X))

    X0 = [dxinit, dyinit]
    dx, dy = minimize(res, X0).x
    return xbase - dx, ybase - dy


def get_data(image):
    lines = pytesseract.image_to_string(
        image[40:256, 80:325, :], config='--psm 6').split("\n")
    lines = [l for l in lines if len(l) > 4]

    """
    for i, l in enumerate(lines):    
        lines[i] = l.replace("°", "")        

    try:
        sog = list(parse.parse("SOG: {:f} Knts", lines[7]))[0]
    except:
        print("FALIED OCR")
        print(lines)
        sog = np.NaN

    try:
        hdg = list(parse.parse("HDG: {:f}", lines[8]))[0]
    except:
        print("FALIED OCR")
        print(lines)
        hdg = np.NaN

    try:
        cog = list(parse.parse("COG: {:f}", lines[9]))[0]
    except:
        print("FALIED OCR")
        print(lines)
        cog = np.NaN

    try:
        rot = list(parse.parse("ROT: {:f}", lines[10]))[0]
    except:
        print("FALIED OCR")
        print(lines)
        rot = np.NaN

    try:
        draught = list(parse.parse("Draught: {:f} m", lines[11]))[0]
    except:
        print("FALIED OCR")
        print(lines)
        draught = np.NaN
    """

    l2 = pytesseract.image_to_string(
        image[115:135, 610:808, :], config='--psm 6').split("\n")
    l2 = [l for l in l2 if len(l) > 4]
    
    """
    h, m, s = list(parse.parse("23 Mar 2021 {:d}:{:d}:{:d} UTC", l2[0]))
    t = 3600*h + 60*m + s
    data = np.array([t, sog, hdg, cog, rot, draught])
    """
    return lines + l2



# %%

vidcap = cv2.VideoCapture('videodata.mp4')
vidcap.set(cv2.CAP_PROP_POS_FRAMES, 5400-1) #5400-1
success, image = vidcap.read()
image = image[0:570, 0:890, [2, 1, 0]]
level = image.dot([0.2989, 0.5871, 0.1140])
cx, dcx = find_xPos(level, 0, 0, True)
cy, dcy = find_yPos(level, 0, 0, True)
x, y = find_boat_pos(level, 0, 0, True)
cX = [[cx, dcx]]
cY = [[cy, dcy]]
XY = [[x, y]]

Data = []
count = 0
while success and count < 2500: #2500
    data = get_data(image)
    Data += [data]

    level = image.dot([0.2989, 0.5871, 0.1140])
    level[0:264, 0:329] = np.NaN
    level[104:199, 579:819] = np.NaN    

    cx, dcx = find_xPos(level, cx, dcx)
    cy, dcy = find_yPos(level, cy, dcy)
    x, y = find_boat_pos(level, x, y)
    cX += [[cx, dcx]]
    cY += [[cy, dcy]]
    XY += [[x, y]]

    success, image = vidcap.read()
    image = image[0:570, 0:890, [2, 1, 0]]
    if count % 10 == 0:
        print('Frame %d: ' % count, success)
        if count % 100 == 0:
            fig = plt.figure(figsize=2*defsize)
            ax = fig.add_subplot(1, 2, 1)
            plt.xlim([x-50, x+50])
            plt.ylim([y-50, y+50])
            plt.imshow(level, cmap="gray")
            plt.gca().invert_yaxis()
            plt.plot(x, y, "2", color='#d62728', mew=4, markersize=40)

            ax = fig.add_subplot(1, 2, 2)
            plt.xlim([x-250, x+250])
            plt.ylim([y-250, y+250])
            plt.imshow(level, cmap="gray")
            plt.gca().invert_yaxis()


            col = np.arange(-10, 11)
            row = np.arange(-10, 11)
            row, col = np.meshgrid(row, col, indexing='ij')
            col = col.ravel()
            row = row.ravel()
            plt.plot(cx + col*dcx, cy + row*dcy, ".")
            plt.plot(cx, cy, "x", markersize=20)

            plt.show()
    count += 1

plt.figure(figsize=2*defsize)
plt.imshow(image)
col = np.arange(-10, 11)
row = np.arange(-10, 11)
row, col = np.meshgrid(row, col, indexing='ij')
col = col.ravel()
row = row.ravel()
plt.plot(cx + col*dcx, cy + row*dcy, ".")
plt.plot(cx, cy, "o")
plt.ylim([0, 570])
plt.xlim([0, 890])
plt.gca().invert_yaxis()

#%%

# cX = np.array(cX)
# cY = np.array(cY)
# XY = np.array(XY)
# Data = np.array(Data)
# 
# np.savetxt("cX.txt", cX, delimiter=",")
# np.savetxt("cY.txt", cY, delimiter=",")
# np.savetxt("XY.txt", XY, delimiter=",")
# 
# f = codecs.open("Data.txt", "w", "utf-8")
# for l in Data:
#     for s in l:
#         f.write(s)
#         f.write(",")
#     f.write("\n")
# f.close()


