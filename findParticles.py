# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 14:00:49 2021

This script is used to import an SEM image, then fit and extract the position of each QD in the image.

@author: Alex
"""
import numpy as np
import skimage.io
import skimage.feature
import os
import matplotlib.pyplot as plt
import time
from pathlib import Path



def FindBlobs(image, ShowBlobs, min_sigma=2, max_sigma=3, threshold=0.01):
    print ("Finding blobs with Laplacian of Gaussian method with parameters:")
    print ("Min_sigma = " , repr(min_sigma))
    print ("Max_sigma = " , repr(max_sigma))
    print ("Threshold = " , repr(threshold))
    
    blobs = skimage.feature.blob_log(image, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=4, 
                                     threshold=threshold, exclude_border = False)
    
    if ShowBlobs:
        print ("plotting")
        fig, ax = plt.subplots()
        skimage.io.imshow(image)
        if len(blobs) != 0:
            plt.plot(blobs[:, 1], blobs[:, 0], 'r.')
        plt.show()
        fig.canvas.draw()
    
    return blobs
    
    

    
