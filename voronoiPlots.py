# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
import matplotlib.cm as cm
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np

def VoronoiColorPlot(InputMatrix,IndexColumn,DataColumn,centroids,label, x_size, y_size):
    x = centroids[:,1]
    y = centroids[:,0] 
    vor = Voronoi(np.vstack((x,y)).T) 

    i=IndexColumn
    d=DataColumn
    dc=InputMatrix[:,d]
    ic=InputMatrix[:,i]
    
    minima = min(dc)
    maxima = max(dc)
    norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='magma')


    fig2,ax2= plt.subplots()

    fig2=voronoi_plot_2d(vor, ax=ax2,show_points=False, show_vertices=False, s=1)
    
    for r in range(len(dc)):
        r0=int(ic[r])
        region = vor.regions[vor.point_region[r0]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            plt.fill(*zip(*polygon), color=mapper.to_rgba(dc[r]))
        
        
    fig2.colorbar(mapper,ax=ax2,label=label)

    #ax2.set_xlim(0,x_size)
    #ax2.set_ylim(0,y_size)

    plt.show()