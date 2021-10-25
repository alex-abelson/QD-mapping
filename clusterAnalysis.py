# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 18:57:11 2021

@author: Alex
"""

import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from pathlib import Path
import os
from sklearn import preprocessing
from sklearn import cluster
from sklearn import metrics

from sklearn.preprocessing import StandardScaler

def PrepareData(DM_Raw):
    DM_Stripped = DM_Raw[:,3:]
    #DM_Scaled = preprocessing.scale(DM_Stripped)
    scaler = StandardScaler()

    DM_Scaled = scaler.fit_transform(DM_Stripped)
    
    return DM_Scaled


def Labels(DesignMatrix, eps, method= "DBSCAN", numBins=3):    
    if method == 'DBSCAN':
        #cluster_out=cluster.DBSCAN(eps=eps, min_samples = 5)
        #labels=cluster_out.fit_predict(DesignMatrix)
        db = cluster.DBSCAN(eps=eps, min_samples=5).fit(DesignMatrix)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        
        print('{} QDs fed into clustering algorithm.'.format(len(DesignMatrix)))
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        print("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(DesignMatrix, labels))

        
        
    elif method == 'KMEANS':
        cluster_out=cluster.KMeans(n_clusters=numBins, algorithm='full',max_iter=500)
        labels=cluster_out.fit_predict(DesignMatrix)
        
    
    return labels


def ClusterFigure(directory, labels,DesignMatrix,image,filename):
    if not os.path.exists(directory + "outputs"):
        os.mkdir(directory + "outputs")
    output_name=directory + "outputs/" +"clusterfig_" + Path(filename).stem

    #fig, ax = plt.subplots()
    #skimage.io.imshow(image)
    
    if len(labels) != 0:
        plt.scatter(DesignMatrix[:,1], DesignMatrix[:,2],  s=10, c=labels)
    plt.savefig(output_name,dpi=1200)
    #plt.show()
    #fig.canvas.draw()
    

    

    
    


    