# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 18:57:11 2021

@author: Alex
"""

import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import os
from sklearn import preprocessing
from sklearn import cluster
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from kneebow.rotor import Rotor

from sklearn.preprocessing import StandardScaler

def PrepareData(DM_Raw):
    DM_Stripped = DM_Raw[:,3:]
    #DM_Scaled = preprocessing.scale(DM_Stripped)
    scaler = StandardScaler()

    DM_Scaled = scaler.fit_transform(DM_Stripped)
    
    return DM_Scaled


def Labels(DesignMatrix, method= "DBSCAN", numBins=3):
    if method == 'DBSCAN':
        
        #plt.close()
        neigh = NearestNeighbors(n_neighbors=12)
        nbrs = neigh.fit(DesignMatrix)
        distances, indices = nbrs.kneighbors(DesignMatrix)
        distances = np.sort(distances, axis=0)
        distances = distances[:,1]
        #plt.plot(distances) #can use these commented out parts to show neighbor spacing
        #plt.show()
        
        index = np.linspace(0,len(distances), num = len(distances))
        
        distances = index,distances
        distances = np.vstack(distances).T  
        
        #This rotor method finds the elbow in distances vs index curve https://bit.ly/3Ck7KUv
        rotor = Rotor()
        rotor.fit_rotate(distances)
        elb = rotor.get_elbow_index()

        eps = distances[elb,1]

        
        db = cluster.DBSCAN(eps=eps, min_samples=30).fit(DesignMatrix)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        
        print("Running cluster analysis with eps={}".format(eps))
        print('{} QDs fed into clustering algorithm.'.format(len(DesignMatrix)))
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        print("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(DesignMatrix, labels))
        

        
    elif method == 'KMEANS':
        cluster_out=cluster.KMeans(n_clusters=numBins, algorithm='full',max_iter=500)
        labels=cluster_out.fit_predict(DesignMatrix)
        
    
    return labels


def ClusterFigure(directory, labels,DesignMatrix,image,filename,BlobSize):
    if not os.path.exists(directory + "outputs"):
        os.mkdir(directory + "outputs")
    output_name=directory + "outputs/" +"clusterfig_" + Path(filename).stem

    
    fig, ax = plt.subplots()
    skimage.io.imshow(image)
    
    if len(labels) != 0:
        labels = labels
        N = np.ptp(labels)+1
        cmap = plt.cm.jet
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)       
        bounds = np.linspace(np.amin(labels)-.5,np.amax(labels)+.5,N+1)
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)        
        
        scat = ax.scatter(DesignMatrix[:,1], DesignMatrix[:,2],  s=BlobSize, c=labels, cmap=cmap,  norm = norm, facecolors='none')
        cb = plt.colorbar(scat, spacing='proportional',ticks=np.arange(np.min(labels),np.max(labels)+1))
      
    plt.savefig(output_name,dpi=600)
    #plt.show()
  
    

def RemoveDefectsFromClustering(directory, filename, labels, matrix):
    SaveLabels = np.array([0])
    DotsToDelete = []
    
    for i,item in enumerate(labels):
        if item not in SaveLabels:
            DotsToDelete.append(i)
        
    MatrixCleaned = np.delete(matrix,DotsToDelete, axis=0)
    
    return MatrixCleaned
    

    


    