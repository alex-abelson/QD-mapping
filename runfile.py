# -*- coding: utf-8 -*-


"""
Author: Alex Abelson, 2021.

QD Mapping Runfile program (runfile.py)

This program has three functions:
    1) Calculate subpixel position of QDs in an image.
    2) Perform a Voronoi decomp. and generate a design matrix.
    3) Classify QDs using design matrix.
    4) Scrub defective particles from resultant design matrix.
"""

#################### Runfile Settings #################################################
CentroidsRun = False
DesignMatrixRun = False
ClusterRun = True
CleanDesignMatrix = True
directory = r'C:/Users/Alex/Desktop/SEM Image Analysis Photobase/PBG z gradient check/'

sizeSet = True
x_size = int(1536)
y_size = int(1024)



#################### Centroids ####################################################
#Set blob detection parameters.
min_sigma = 2.8
max_sigma = 3.0
threshold = 0.005
ShowBlobs = True

#Subpixel Fitting
ShowSubpixels = False
conversion = 1.48 #pixels/nm
spacing = 6.5*conversion #sets the size of the window for Gaussian fitting

################### DesignMatrix ##################################################
buffer = 10 #number of pixels on edge of image to remove QDs
ShowVoronoi = False
#Color Voronoi Plot
DataColumn=3
ColorLabel="NNs"

#################### Clustering ###################################################
ShowCluster = True
eps=0.85
#################################################################################


import numpy as np
import os
import time
from pathlib import Path
import findParticles as fp
import designMatrix as dm
import clusterAnalysis as ca
import subpixel as sp
import utilities as ut
import voronoiPlots as vp


def Particles(filename, x_size, y_size, ShowBlobs, min_sigma, max_sigma, threshold):
    """This function loads an SEM image, does the LoG blob fitting, and returns 
    a x,y,r array with QD position and the radius (r) used in the fit."""

    #Load image as a 64-bit float. Crop data bar.
    image = ut.LoadImage(filename, x_size, y_size)
    #Run blob detection. Display results for verification...
    blobs_out = fp.FindBlobs(image, ShowBlobs, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
    print ("Found {} QDs with blob finder.".format(len(blobs_out)))
   
    return blobs_out

def Subpixel(directory, filename,centroids,x_size, y_size, conversion, spacing, ShowSubpixels):
    """Subpixel centroid detection algorithm. Takes in centroid position from blob
    detector, then does 2-D Gaussian fitting of each QD. Returns the new x,y QD
    positions"""
    
    image = ut.LoadImage(filename, x_size, y_size)  #Load image.
    x = centroids[:,0]                             #Define x and y QD pos array.
    y = centroids[:,1]   

    print ("Performing 2D gaussian fit to all peaks...")

    x_SP=[]
    y_SP=[]

    shift = spacing/2 
    unfit_particles = 0
    edge_particles = 0
    
    time_init = time.time()
    for i in range(len(x)):
        #print ("Fitting particle {} of {}".format(i+1, len(x)))        
        xcurr,ycurr = int(x[i]),int(y[i])
        if not sp.on_edge(xcurr,ycurr,shift,image):
            xSPcurr,ySPcurr, fit= sp.xySP(xcurr,ycurr,spacing,image, ShowSubpixels)
            if not sp.on_edge(xSPcurr,ySPcurr,shift,image):
                x_SP.append(xSPcurr)
                y_SP.append(ySPcurr)
                unfit_particles += fit
            else:
                edge_particles += 1
        else:
            edge_particles += 1            
    time_tot = time.time() - time_init
    
    # Shift centers
    x_SP, y_SP = np.array(x_SP), np.array(y_SP)

    print ("Done.\n{} total particles.".format(len(x)))
    print ("{} edge particles were discarded.".format(edge_particles))
    print ("{} particles could not be fit.".format(unfit_particles))
    print ("Process took {} seconds.".format(repr(int(time_tot))))
    
    #Save centroid positions.
    if not os.path.exists(directory + "outputs"):
        os.mkdir(directory + "outputs")
    output_name=directory + "outputs/"+"centroid_subpixel_" + Path(filename).stem 
    new_centroids = np.vstack((x_SP, y_SP)).T
    np.savez(output_name, new_centroids=new_centroids)
    #np.savetxt(output_name, new_centroids, delimiter = ",")

    return new_centroids

def MakeDesignMatrix(directory, filename,centroids,x_size, y_size, ShowVoronoi):
    """This function takes in the x-y position of QDs, performs a Voronoi
    decomposition, extracts the structural parameters, then compiles it into and 
    returns the design matrix used for clustering."""
    
    #Load image and centroids.
    image = ut.LoadImage(filename, x_size, y_size) 
    
    #Perform Voronoi decomposition
    print ("Performing Voronoi decomposition...")
    x_pos,y_pos,vor = dm.RunVoronoi(centroids, image, filename, ShowVoronoi)
    
    #Generate the structural parameters. These are [n,1] arrays.
    print ("Calculating structural parameters...")
    IndexRef, NNList_out, NNList = dm.NNList(vor)
    avgDistance, stdDistance, Distances, NNDiff = dm.NNDistance(vor, NNList)
    Psi4, Psi6, argListSin_6, argListCos_6, argListSin_4, argListCos_4 = dm.BondOrder(vor, NNList)

    #Build design matrix stack from the structural [n,1] arrays above.
    DesignMatrixTemp = (IndexRef,       #Particle ID
                        x_pos,          #x-position of QD
                        y_pos,          #y-position of QD
                        NNList_out,     #Number of nearest neighbors
                        avgDistance,    #Average inter-QD distance
                        Psi4,           #psi-4
                        Psi6,           #psi-6
                        argListSin_4,   #sin of complex arg. of psi-4
                        argListCos_4)   #cos of complex arg. of psi-4
    
    DesignMatrixStack = np.vstack(DesignMatrixTemp).T       
    
    #Delete edge particles  
    toDelete = dm.RemoveEdge(vor,y_size,x_size, buffer)    
    DesignMatrix = np.delete(DesignMatrixStack,toDelete,0)    
    NNDiff = np.delete(NNDiff,toDelete,0)
    
    #Save Design Matrix
    if not os.path.exists(directory + "outputs"):
        os.mkdir(directory + "outputs")
    output_name=directory + "outputs/" +"designmatrix_" + Path(filename).stem   
    header = "Image ID = " + Path(filename).stem + " Index, X, Y, Nearest Neighbors, Avg NN Distance, Psi4, Psi6, ArgSin(Psi4), ArgCos(Psi4)" 
    np.savez(output_name, DesignMatrix=DesignMatrix)
    np.savetxt(output_name + ".txt",DesignMatrix,delimiter = ',', header = header)
    
    output_name=directory + "outputs/" +"distances_" + Path(filename).stem 
    np.savetxt(output_name + ".txt",Distances, delimiter = ',')
    
    output_name=directory + "outputs/" +"distance_diffs_" + Path(filename).stem 
    np.savetxt(output_name + ".txt",NNDiff, delimiter = ',')
    print ("Files saved containing {} QDs.".format(len(DesignMatrix)))
    
    return DesignMatrix
    

def Clustering(directory, filename, DesignMatrixTemp, eps, x_size, y_size, ShowCluster):
    """This function takes in the design matrix and performs a clustering
    analysis to classify the QDs."""

    #Load image and design matrix.
    image = ut.LoadImage(filename, x_size, y_size)

    
    #Prepare and cluster design matrix data.
    DesignMatrixCluster = ca.PrepareData(DesignMatrixTemp) 
    labels = ca.Labels(DesignMatrixCluster, eps)
    
    
    if ShowCluster:     #Generate cluster figure.
        ca.ClusterFigure(directory, labels,DesignMatrixTemp,image,filename)   
    
    #Save clustering labels.
    if not os.path.exists(directory + "outputs"):
        os.mkdir(directory + "outputs")
    output_name=directory + "outputs/" +"clustering_" + Path(filename).stem + '.txt'   
    np.savetxt(output_name,labels,delimiter = ',')
    np.savez(output_name, labels=labels)
    
    
if __name__ == "__main__":
    
    for filename in os.listdir(directory):
        if filename.endswith(".tif"):
            print ("Processing image: " + Path(filename).stem)
            filename = directory + filename
            
            if sizeSet:
            
                x_size = ut.GetDimensions(filename)[0]
                y_size = ut.GetDimensions(filename)[1]
            
            print (x_size,y_size)
            
            if CentroidsRun:
                centroids = Particles(filename, x_size, y_size, ShowBlobs, min_sigma, max_sigma, threshold)
                new_centroids = Subpixel(directory, filename, centroids, x_size, y_size, conversion, spacing, ShowSubpixels)
                print(1 * "\n")
                
            if DesignMatrixRun:
                #Load Centroids npz.
                output_name = directory + "outputs/"+"centroid_subpixel_" + Path(filename).stem 
                new_centroids_file = np.load(output_name + ".npz")
                new_centroids = new_centroids_file['new_centroids']
                designmatrix = MakeDesignMatrix(directory, filename, new_centroids, x_size, y_size, ShowVoronoi)
                print(1 * "\n")
                
            if ClusterRun:
                #Load Design Matrix
                output_name = directory + "outputs/" + "designmatrix_" + Path(filename).stem
                designmatrix_file = np.load(output_name + ".npz")
                designmatrix = designmatrix_file['DesignMatrix'] 
                #vp.VoronoiColorPlot(designmatrix,0,DataColumn,new_centroids,ColorLabel,x_size,y_size)
                Clustering(directory, filename, designmatrix, eps, x_size, y_size, ShowCluster)
            
            if CleanDesignMatrix:
                output_name = directory + "outputs/" + "designmatrix_" + Path(filename).stem
                designmatrix_file = np.load(output_name + ".npz")
                designmatrix = designmatrix_file['DesignMatrix']
                
                output_name=directory + "outputs/" +"clustering_" + Path(filename).stem + '.txt'  
                cluster_file = np.load(output_name + ".npz")
                labels = cluster_file['labels']
                
                ca.RemoveDefectsFromClustering(directory, filename, labels, designmatrix)
                
                print(1 * "\n")
        else:
            continue
        print(2 * "\n")  
    
    print ("Done.")
    
    
"""time_init=time.time()
time_tot = time.time() - time_init
print ("Calculated structural parameters in {} seconds.".format(repr(int(time_tot))))"""
        
    

    

           