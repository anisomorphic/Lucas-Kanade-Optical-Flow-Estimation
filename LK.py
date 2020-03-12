# Michael Harris
# CAP 4453 - Robot Vision - Spring 2019
# PA2 - Lucas Kanade optical flow (non multi-scale version)

import cv2
import numpy as np
from matplotlib import pyplot as plot

# lucas kanade, operates between two images (F1, F2)
def LK(F1, F2):
    win = 5
    shape = F1.shape

    # prepare arrays
    Ix = np.ones(shape)
    Iy = np.ones(shape)
    It = np.ones(shape)
    Ix[1:-1, 1:-1] = (F1[1:-1, 2:] - F1[1:-1, :-2]) / 2
    Iy[1:-1, 1:-1] = (F1[2:, 1:-1] - F1[:-2, 1:-1]) / 2
    It[1:-1, 1:-1] = F1[1:-1, 1:-1] - F2[1:-1, 1:-1]

    # gaussian blur
    params = np.zeros(shape + (5,))
    params[..., 0] = cv2.GaussianBlur(Ix * Ix, (win, win), 1) # xx
    params[..., 1] = cv2.GaussianBlur(Iy * Iy, (win, win), 1) # yy
    params[..., 2] = cv2.GaussianBlur(Ix * Iy, (win, win), 1) # xy
    params[..., 3] = cv2.GaussianBlur(Ix * It, (win, win), 1) # xt
    params[..., 4] = cv2.GaussianBlur(Iy * It, (win, win), 1) # yt

    # prepare the window, 2*sigma+1, default sigma is 5
    cumulative = np.cumsum(np.cumsum(params, axis=0), axis=1)
    window = (cumulative[2 * win + 1:, 2 * win + 1:] -
              cumulative[2 * win + 1:, :-1 - 2 * win] -
              cumulative[:-1 - 2 * win, 2 * win + 1:] +
              cumulative[:-1 - 2 * win, :-1 - 2 * win])

    # prepare vector arrays
    u = np.zeros(shape)
    v = np.zeros(shape)

    # setup matrices
    Ixx = window[..., 0]
    Iyy = window[..., 1]
    Ixy = window[..., 2]
    Ixt = -window[..., 3]
    Iyt = -window[..., 4]

    # matrix ops to determine vector strength and orientation
    det = Ixx * Iyy - Ixy ** 2
    _u = Iyy * (-Ixt) + (-Ixy) * (-Iyt)
    _v = (-Ixy) * (-Ixt) + Ixx * (-Iyt)
    flow_x = np.where(det != 0, _u / det, 0)
    flow_y = np.where(det != 0, _v / det, 0)

    u[win + 1: -1 - win, win + 1: -1 - win] = flow_x[:-1, :-1]
    v[win + 1: -1 - win, win + 1: -1 - win] = flow_y[:-1, :-1]

    return u, v

# produces a map, given u and v
def flow_map(u, v):
    win = 8
    flow_map = np.ones(u.shape)

    # loop through and apply a scalar to values, draw line at x,y
    for y in range(flow_map.shape[0]):
        for x in range(flow_map.shape[1]):
            if y % win == 0 and x % win == 0:
                dy = 15 * int (v[y, x])
                dx = 15 * int (u[y, x])

                if dy > 0 or dx > 0:
                    cv2.arrowedLine(flow_map, (x, y), (x + dx, y + dy), 128, 1)

    return flow_map


# open images
b1 = cv2.cvtColor(cv2.imread('basketball1.png'), cv2.COLOR_RGB2GRAY)
b2 = cv2.cvtColor(cv2.imread('basketball2.png'), cv2.COLOR_RGB2GRAY)

# run algorithms
u1, v1 = LK(b1, b2)
flow = flow_map(u1,v1)

mask = b2 + flow

# prepare plots, show
plot.subplot(2,2,1),plot.imshow(b1, cmap='gray'),plot.title('basketball1')
plot.xticks([]), plot.yticks([]) #remove x and y values from each 'graph'
plot.subplot(2,2,2),plot.imshow(b2, cmap='gray'),plot.title('basketball2')
plot.xticks([]), plot.yticks([])
plot.subplot(2,2,3),plot.imshow(flow, cmap='gray'),plot.title('flow map')
plot.xticks([]), plot.yticks([])
plot.subplot(2,2,4),plot.imshow(mask, cmap='gray'),plot.title('overlay')
plot.xticks([]), plot.yticks([])

plot.show()
