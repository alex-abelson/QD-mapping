# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 15:11:27 2021

Loads an image and an .npz file containing the centroid locations of each QD in the image.

@author: Alex
"""

from utilities import find_index,angle_between
from scipy.spatial import Voronoi,ConvexHull,voronoi_plot_2d
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import time
from pathlib import Path
import os



def NNList(vor): #Returns the nearest neighbor particles for each particle.
    IndexRef = np.zeros(len(vor.points))
    IndexRef = [i for i in range(len(vor.points))]    
    NNList = []    
    for particle in range(len(vor.points)):
        
        IndicesX = list(np.flatnonzero(vor.ridge_points[:,0]==particle))
        PointsX = [vor.ridge_points[item,1] for item in IndicesX]
            
        IndicesY = list(np.flatnonzero(vor.ridge_points[:,1]==particle))
        PointsY = [vor.ridge_points[item,0] for item in IndicesY]
            
        NNList.append(np.array(PointsX + PointsY))     
        NNList_out = np.asarray([len(item) for item in NNList])
        
    return IndexRef, NNList_out, NNList
        
    
def NNDistance(vor,NNList): #Calculates the avg and std distance between Particle i and its NNs.

    avgDistance = []
    stdDistance = []
    Distances = []
    
    for i,i_xy in enumerate(vor.points):
        DistanceDummy = []

        for i_NN in NNList[i]:
            DistanceDummy.append(np.linalg.norm(i_xy-vor.points[i_NN]))
        DistanceDummyNP=np.asarray(DistanceDummy)
            
        if len(DistanceDummyNP!=0):
            avgDistance.append(np.mean(DistanceDummyNP))
            stdDistance.append(np.std(DistanceDummyNP))
        else:
            avgDistance.append(0)
            stdDistance.append(0)
        if len(DistanceDummyNP)==6:
            Distances.append(DistanceDummyNP)
        else:
            Distances.append([0,0,0,0,0,0])

    return avgDistance, stdDistance, Distances


def BondOrder(vor, NNList):
    
    Psi4 = []
    Psi6 = []
    argListSin_6 = []
    argListCos_6 = []   
    argListSin_4 = []
    argListCos_4 = []
    
    arr = vor.ridge_points[:,:]
    arr = arr.astype('int64')
    RidgeList = []
    
    for particle in range(len(vor.points)):
        RidgeListRow=[]
    
        for neighbor in NNList[particle]:
        
            x1=np.array([neighbor,particle],dtype='int64')
            x2=np.array([particle,neighbor],dtype='int64')
            
            dummy = find_index(arr,x1)               
            if dummy.size == 0:
                dummy=find_index(arr,x2)
                
            RidgeListRow.append(dummy[0])
        RidgeList.append(RidgeListRow)
        
    for ParticleIndex,ParticleCoordinate in enumerate(vor.points):
            
        RidgeLengthRow = []
        AngleListRow = []
        
        for NeighborCounter,NeighborIndex in enumerate(NNList[ParticleIndex]):
            
            n =np.array(vor.vertices[vor.ridge_vertices[RidgeList[ParticleIndex][NeighborCounter]]])
            
            ridge_length_temp=np.array([n[1,0]-n[0,0],n[1,1]-n[0,1]])                
            RidgeLengthRow.append(np.linalg.norm(ridge_length_temp))
            
            angle_temp=angle_between(vor.points[ParticleIndex]-vor.points[NeighborIndex], np.array([1,0]))
            AngleListRow.append(angle_temp)
            
            

            psi_6_sum_temp = 0
            psi_4_sum_temp = 0

        for idx,angle in enumerate(AngleListRow):

            psi_6_temp = np.exp(6*1j*AngleListRow[idx])
            psi_4_temp = np.exp(4*1j*AngleListRow[idx])
            ridge_length_temp = RidgeLengthRow[idx]
            
            psi_6_sum_temp += psi_6_temp*ridge_length_temp
            psi_4_sum_temp += psi_4_temp*ridge_length_temp


        arg_6 = np.angle(psi_6_sum_temp/(np.sum(RidgeLengthRow)))
        arg_4 = np.angle(psi_4_sum_temp/(np.sum(RidgeLengthRow)))

        Psi6.append(np.abs(psi_6_sum_temp/np.sum(RidgeLengthRow)))
        Psi4.append(np.abs(psi_4_sum_temp/np.sum(RidgeLengthRow)))

        argListSin_6.append(np.cos(arg_6))
        argListCos_6.append(np.sin(arg_6))

        argListSin_4.append(np.cos(arg_4))
        argListCos_4.append(np.sin(arg_4))
                    
    Psi4 = np.asarray(Psi4)
    Psi6 = np.asarray(Psi6)
    
    argListSin_6 = np.asarray(argListSin_6)
    argListCos_6 = np.asarray(argListCos_6)
    
    argListSin_4 = np.asarray(argListSin_4)
    argListCos_4 = np.asarray(argListCos_4)
    
    return Psi4, Psi6, argListSin_6, argListCos_6, argListSin_4, argListCos_4


def CalcArea(vor):
    Areas = np.zeros(vor.npoints)
    
    for i,reg_num in enumerate(vor.point_region):
        indices = vor.regionsp[reg_num]
        if -1 in indices:
            Areas[i] = np.inf
        else:
            Areas[i] = ConvexHull(vor.vertices[indices].volume)
            
    return Areas


def RemoveEdge(vor,xsize,ysize, buffer):
    ToDelete = []
    
    for i,x in enumerate(vor.points):  
        for item in (vor.regions[vor.point_region[i]]):     
            if item==-1 or vor.vertices[item][0]<buffer or vor.vertices[item][1]<buffer or vor.vertices[item][0]>(ysize-buffer) or vor.vertices[item][1]>(xsize-buffer):
                ToDelete.append(i)
                break

    return ToDelete    

def RunVoronoi(centroids, image, filename, ShowVoronoi):
    #centroids = np.load(centroid_file)
    x = centroids[:,1]
    y = centroids[:,0] 

    vor = Voronoi(np.vstack((x,y)).T)    
    if ShowVoronoi:
        
        if not os.path.exists("outputs"):
            os.mkdir("outputs")
        output_name="outputs/"+"voronoifig_" + Path(filename).stem
        
        fig, ax = plt.subplots()
        skimage.io.imshow(image)
        fig = voronoi_plot_2d(vor,ax=ax, show_vertices=False, linewidth = 2, point_size = 2)
        plt.savefig(output_name,dpi=1200)
        plt.show()
        
    return x,y,vor

    
    
    
    
    