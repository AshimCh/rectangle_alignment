import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# Reading the image
im = cv2.imread('R2.png')

# Converting image to grayscale
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Thresholding and getting contours from the image
for i, c in enumerate(contours):
    areaContour=cv2.contourArea(c)
    if areaContour<2000 or 100000<areaContour:
        continue
    cv2.drawContours(im,contours,i,(68,214,44),4)

# Getting the center
    M = cv2.moments(c)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = c - [cx, cy]

    def cart2pol(x, y):
        theta = np.arctan2(y, x)
        rho = np.hypot(x, y)
        return theta, rho


    def pol2cart(theta, rho):
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return x, y


    def rotate_contour(c, angle):
        M = cv2.moments(c)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        cnt_norm = c - [cx, cy]
        
        coordinates = cnt_norm[:, 0, :]
        xs, ys = coordinates[:, 0], coordinates[:, 1]
        thetas, rhos = cart2pol(xs, ys)
        
        thetas = np.rad2deg(thetas)
        thetas = (thetas + angle) % 360
        thetas = np.deg2rad(thetas)
        
        xs, ys = pol2cart(thetas, rhos)
        
        cnt_norm[:, 0, 0] = xs
        cnt_norm[:, 0, 1] = ys

        cnt_rotated = cnt_norm + [cx, cy]
        cnt_rotated = cnt_rotated.astype(np.int32)

        return cnt_rotated

    cnt_rotated = rotate_contour(contours[i], 60)
    cv2.drawContours(im, contours, 0, (255, 0, 0), 3)
    cv2.drawContours(im, [cnt_rotated], 0, (0, 255, 0), 3)

cv2.imshow('The real image',imgray)
cv2.imshow('Rotate',im)
cv2.waitKey(0)

