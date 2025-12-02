# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 09:34:14 2022

@author: MMBM_CAROLINE
"""
import cv2
import os
import numpy as np
import pandas as pd
import time
import modules.functions as functions
import modules.morphology as morphology
from skimage.util import img_as_ubyte 
from skimage import feature
from scipy.spatial.distance import cdist


def fill( # Fill holes in closed edges
        edges_closed
    ):

    contours, _  = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # RETR_TREE    
    binMaskFilled = np.zeros_like(edges_closed)
    binMaskFilled = cv2.drawContours(binMaskFilled, contours, -1, 255, thickness = cv2.FILLED) 

    return binMaskFilled

def closing( # Close edges (dilation followed by erosion)
        edges,
        kernel
    ):

    kernel = np.ones(kernel, np.uint8)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return edges_closed

def getMask( # Get a mask of the image using Canny edge detector
        image, 
        gaussianFilterSD, # Standard deviation of the Gaussian filter to blur the image 
        dilationKernel,
        threshMin, # Low threshold for the canny edge algorithm
        threshMax # High threshold for the canny edge algorithm
    ):

    # Detect edges with Canny algorithm
    edges = img_as_ubyte(feature.canny(image, sigma = gaussianFilterSD, low_threshold = threshMin, high_threshold = threshMax)) # Better than cv2.Canny that support only 8bits images
     
    # Dillate and erose the edges to obtain closed shapes
    edges_closed = closing(edges, dilationKernel)
    
    # Fill the holes inside the close shapes
    binMask = fill(edges_closed)
    # binMask = edges_closed
   
    return binMask, edges, edges_closed

def filterContours( # Filter the contour to detect objects of interest with size, circularity ang grey intensity criterions
        image,
        contours, 
        minGrayLevel,
        maxGrayLevel, 
        minArea, # Remove small artefacts
        maxArea, # Remove big area
        minCircularity ,
        minAspectRatio
    ):
    
    circularity = morphology.getCirculartity(contours)
    aspect_ratio = morphology.getAspectRatio(contours)

    objects_filtArea = []
    objects = []
    
    # Filter contours, in 2 steps because getMeanGreyLevel takes too much time to do it for each contour
    for i in range(len(contours)): # Filter by size
        if  cv2.contourArea(contours[i]) < maxArea and cv2.contourArea(contours[i]) > minArea and circularity[i] > minCircularity and aspect_ratio[i] > minAspectRatio :
            objects_filtArea.append(contours[i])
            
    for i in range(len(objects_filtArea)): # Filter by grey intensity
        meanGrayLevel, __, __ =  morphology.getMeanGreyLevel(image, objects_filtArea)
        if meanGrayLevel[i] < maxGrayLevel and meanGrayLevel[i] > minGrayLevel:
            objects.append(objects_filtArea[i])

    return objects

def removeSameObjects( # Among a list of contours, identify the objects with closed centroids (same objects, but inner/outer contour) and keep the outer contour)
        objects
    ):

    objects_select = []
    centroids = morphology.getCentroid(objects)

    if len(objects) == 0: # If no object detected, skip the process
        return objects_select
    
    # Compute the distances between all the centroids
    distance_matrix = cdist(centroids, centroids) 

    # Group the index of closed objects (distance < 10) 
    groups = []
    
    for i in range(len(centroids)):
        group_exist = False
    
        for group in groups:
            if any(distance_matrix[i, j] < 20 for j in group):
                group.append(i)
                group_exist = True
                break
        
        if not group_exist:
            groups.append([i])

    # Keep the outer contour in each group (the bigest object)
    for group in groups:
        area_max = 0
        object_select = None
    
        for i in group:
            area = cv2.contourArea(objects[i])
            if area > area_max:
                area_max = area
                object_select = objects[i]
        
        objects_select.append(object_select)
        
    return objects_select

def getObjects( # Detect the objects of interest in the image
        image,
        binMask, 
        minGrayLevel,
        maxGrayLevel,
        minArea, 
        maxArea, 
        minCircularity,
        minAspectRatio
    ):
    
    # Deteact ALL the objects on the image binary mask
    contours, __ = cv2.findContours(binMask, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    
    # Select the objects of interests thanks to their size and intensity
    objects = filterContours(image, contours, minGrayLevel, maxGrayLevel, minArea, maxArea, minCircularity, minAspectRatio)
    # objects = removeSameObjects(objects)

    # Measure several morphological parameters for each object
    objects_props,mask_greyLevel, mask_backgroundGreyLevel, masks = morphology.propertiesContours(image, objects) 

    return objects, objects_props,mask_greyLevel, mask_backgroundGreyLevel, masks

def debug_getObjects( # Draw on the initial image the contours of the detected objects
        image, 
        objects, 
        objects_props
    ):
    
    image = functions.convertToRGB(image, normalize = 256/16) # Bright image
    
    # Draw the contours of the objects
    objects_img = cv2.drawContours(image, objects, -1, (0,255,0), 1) 
   
    # Label the objects
    for i in objects_props.index:  
        cv2.putText(objects_img,"%s" % str(i), objects_props['Centroid'][i],
                fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (255, 0, 255), thickness = 2
                )
        
    return objects_img


