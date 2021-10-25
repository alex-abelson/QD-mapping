import numpy as np
import matplotlib.pyplot as plt
import os
import time
import argparse
import skimage.io
import scipy.optimize as opt


def filter_dot(x0,y0,size,image):
    x0=int(x0)
    y0=int(y0)
    rad = int(np.ceil(size/2))  
    xpixels = np.shape(image)[0]
    ypixels = np.shape(image)[1]
    xmin = int(x0-rad)
    xmax = int(x0+rad)
    ymin = int(y0-rad)
    ymax = int(y0+rad)

    smallImage = image[xmin:xmax, ymin:ymax]

    return smallImage, xmin, ymin


	
def display(smallImage, xcenter, ycenter):
    fig, ax = plt.subplots()
    plt.imshow(smallImage, cmap="gray")
    plt.plot(xcenter,ycenter,'r.')
    plt.show()
    fig.canvas.draw()
    

def gauss2d(A, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    (x, y) = A
    x0, y0 = float(x0), float(y0)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + np.abs(amplitude)*np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2) ))
    return  g.ravel()

def on_edge(x0,y0,rad,image):
    xpixels = np.shape(image)[0]
    ypixels = np.shape(image)[1]
    xmin, xmax = x0-rad, x0+rad
    ymin, ymax = y0-rad, y0+rad
    if (xmin <= 0) or (xmax >= xpixels) or (ymin <= 0) or (ymax >= ypixels):
        return True
    else:
        return False
	

def xySP(x0, y0, spacing, image, plot):
    # Get smaller image for faster processing
    smallImage, xshift, yshift = filter_dot(x0,y0,spacing,image)
    # Get new centers
    x0_smallIm, y0_smallIm = x0-xshift, y0-yshift


    # Define mesh for input values and initial guess
    #(xs,ys) = 
    A = np.meshgrid(range(np.shape(smallImage)[1]),range(np.shape(smallImage)[0]))
    initial_guess = (smallImage[int(y0_smallIm),int(x0_smallIm)], y0_smallIm, x0_smallIm, spacing/4.0, spacing/4.0,0,0)
    
    # Set all values outside circle of radius spacing/2 to min value in a small image to account for adjacent particles
    baseline = smallImage.min()
    for i in range(np.shape(smallImage)[0]):
        for j in range(np.shape(smallImage)[1]):
            if (x0_smallIm - i)**2 + (y0_smallIm - j)**2 > (spacing/2.0)**2:
                smallImage[i,j] = baseline
    # Perform fit and pull out centers
    try:
        popt, pcov = opt.curve_fit(gauss2d, A, smallImage.ravel(), p0=initial_guess)
    except RuntimeError:
        print ("Particle could not be fit to a 2D gaussian.  Returning original centroid.")
        return x0, y0, 1
    
    x_SP, y_SP = popt[2]+xshift, popt[1]+yshift
    
    
    

    
    # Plotting for troubleshooting
    if plot:
        (x,y)=A
        data_fitted = gauss2d(A, *popt)
        fig,ax=plt.subplots(1,1)
        ax.imshow(smallImage)
        ax.contour(x,y,data_fitted.reshape(x.shape),8,colors='w')
        plt.show()
        

    return x_SP, y_SP, 0

