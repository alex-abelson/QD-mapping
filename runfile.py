# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 22:29:15 2021

@author: Alex
"""
import numpy as np
import os
import time
from pathlib import Path
import findParticles as fp
import designMatrix as dm
import clusterAnalysis as ca
import skimage.io
import subpixel as sp
import utilities as ut
import voronoiPlots as vp


#################### Run Settings #################################################
CentroidsRun = True
DesignMatrixRun = True
ClusterRun = True
directory = r'C:/Users/Alex/Desktop/SEM Image Analysis Photobase/FIGURE 2/TBAH 100 uM/'
x_size = int(1536)
y_size = int(1024)

#################### Centroids ####################################################
#Set blob detection parameters.
min_sigma = 3.0
max_sigma = 3.4
threshold = 0.02
ShowBlobs = False

#Subpixel Fitting
ShowSubpixels = False
conversion = 1.85 #pixels/nm
spacing = 8.6*conversion #sets the size of the window for Gaussian fitting

################### DesignMatrix ##################################################
buffer = 15 #number of pixels on edge of image to remove QDs
ShowVoronoi = False
#Color Voronoi Plot
DataColumn=3
ColorLabel="NNs"

#################### Clustering ###################################################
ShowCluster = True
eps=0.78
#################################################################################


def Particles(filename, x_size, y_size, ShowBlobs, min_sigma, max_sigma, threshold):
    #if not os.path.exists("outputs"):
        #os.mkdir("outputs")
    #output_name="outputs/"+"centroids_" + Path(filename).stem
    
    #Load image as a 64-bit float. Crop data bar.
    image = ut.LoadImage(filename, x_size, y_size)
    

    
    #Run blob detection. Display results for verification...
    blobs_out = fp.FindBlobs(image, ShowBlobs, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
    print ("Found {} QDs with blob finder.".format(len(blobs_out)))
    #np.savez(output_name,x=np.array(blobs_out[:,1]), y=np.array(blobs_out[:,0]))
    
    return blobs_out


def MakeDesignMatrix(directory, filename,centroids,x_size, y_size, ShowVoronoi):
    #Load image and centroids.
    image = ut.LoadImage(filename, x_size, y_size)
    
    #Perform Voronoi decomposition
    print ("Performing Voronoi decomposition...")
    x_pos,y_pos,vor = dm.RunVoronoi(centroids, image, filename, ShowVoronoi)
    
    #Generate Voronoi output arrays.
    print ("Calculating structural parameters...")
    time_init=time.time()
    
    #Generate the structural parameters. These are [n,1] arrays.
    IndexRef, NNList_out, NNList = dm.NNList(vor)
    avgDistance, stdDistance, Distances = dm.NNDistance(vor, NNList)
    Psi4, Psi6, argListSin_6, argListCos_6, argListSin_4, argListCos_4 = dm.BondOrder(vor, NNList)
    
    time_tot = time.time() - time_init
    print ("Calculated structural parameters in {} seconds.".format(repr(int(time_tot))))
    
    
    #Build design matrix stack from the structural [n,1] arrays above.
    DesignMatrixTemp = (IndexRef,
                        x_pos,
                        y_pos,
                        NNList_out,
                        avgDistance,
                        Psi4,
                        Psi6,
                        argListSin_4,
                        argListCos_4)   
    DesignMatrixStack = np.vstack(DesignMatrixTemp).T
    
    #Delete edge particles  
    toDelete = dm.RemoveEdge(vor,y_size,x_size, buffer)    
    DesignMatrix = np.delete(DesignMatrixStack,toDelete,0)
    
    #Save Design Matrix
    if not os.path.exists(directory + "outputs"):
        os.mkdir(directory + "outputs")
    output_name=directory + "outputs/" +"designmatrix_" + Path(filename).stem
    
    header = "Image ID = " + Path(filename).stem + " Index, X, Y, Nearest Neighbors, Avg NN Distance, Psi4, Psi6, ArgSin(Psi4), ArgCos(Psi4)" 
    np.savez(output_name, DesignMatrix=DesignMatrix)
    np.savetxt(output_name + ".txt",DesignMatrix,delimiter = ',', header = header)
    print ("Files saved containing {} QDs.".format(len(DesignMatrix)))
    
    return DesignMatrix
    

def Clustering(directory, filename, DesignMatrixTemp, eps, x_size, y_size, ShowCluster):

    
    print ("Loading image and design matrix...")
    #Load image and design matrix.
    image = ut.LoadImage(filename, x_size, y_size)

    print ("Running cluster analysis with eps={}".format(eps))
    #Prepare and cluster design matrix data.
    DesignMatrixCluster = ca.PrepareData(DesignMatrixTemp) 
    labels = ca.Labels(DesignMatrixCluster, eps)
    

    #Append clusters to figure.
    if ShowCluster:
        ca.ClusterFigure(directory, labels,DesignMatrixTemp,image,filename)   
    
    #Save clustering labels.
    if not os.path.exists(directory + "outputs"):
        os.mkdir(directory + "outputs")
    output_name=directory + "outputs/" +"clustering_" + Path(filename).stem + '.txt'
    
    np.savetxt(output_name,labels,delimiter = ',')
    np.savez(output_name, labels=labels)
    
def Subpixel(directory, filename,centroids,x_size, y_size, conversion, spacing, ShowSubpixels):
    
    if not os.path.exists(directory + "outputs"):
        os.mkdir(directory + "outputs")
    output_name=directory + "outputs/"+"centroid_subpixel_" + Path(filename).stem 
    
    image = ut.LoadImage(filename, x_size, y_size)
    x = centroids[:,0]
    y = centroids[:,1]   

    print ("Performing 2D gaussian fit to all peaks...")
    time_init = time.time()
    x_SP=[]
    y_SP=[]

    shift = spacing/2 
    unfit_particles = 0
    edge_particles = 0
    
    
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
    
    new_centroids = np.vstack((x_SP, y_SP)).T
    np.savez(output_name, new_centroids=new_centroids)
    #np.savetxt(output_name, new_centroids, delimiter = ",")

    return new_centroids

if __name__ == "__main__":
    
    for filename in os.listdir(directory):
        if filename.endswith(".tif"):
            print ("Processing image: " + Path(filename).stem)
            filename = directory + filename
            
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
                print(1 * "\n")
        else:
            continue
        print(2 * "\n")      
    print ("Done.")
    
    
    
    

    

           