# -*- coding: utf-8 -*-

#This script contains some basic utility functions used throughout the image analysis software.


import numpy as np
import skimage.io
import matplotlib.pyplot as plt

def LoadImage(fileName,x_size, y_size):
    y_size=int(y_size)
    x_size=int(x_size)
    image = skimage.io.imread(fileName)[0:y_size,0:x_size] #import image
    return image

def asvoid(arr): #Converts an ND-array to a 1D array. https://stackoverflow.com/a/16216866/190597
    arr = np.ascontiguousarray(arr)
    return arr.view(np.dtype((np.void,arr.dtype.itemsize * arr.shape[-1])))

def find_index(arr, x):
    arr_as1d = asvoid(arr)    
    x = asvoid(x)
    return np.nonzero(arr_as1d == x)[0]

def unit_vector(vector): #Returns unit vector.
    return vector/np.linalg.norm(vector)

def angle_between(v1, v2): #Returns the angle in radians between inputs v1 and v2
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    minor = np.linalg.det(np.stack((v1_u[-2:], v2_u[-2:])))
    if minor == 0:
        pass#raise NotImplementedError('Too odd vectors =(')
    return np.sign(minor)*np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def angle_betweenU(v1, v2): #Not sure whether this one or the function above is used.
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    minor = np.linalg.det(np.stack((v1_u[-2:], v2_u[-2:])))

    if minor == 0:
        pass
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def PlotImage(image):
    skimage.io.imshow(image)
    plt.show()
    

def GetDimensions(fileName):
    image = skimage.io.imread(fileName)[:,:] #import image
    return image.shape