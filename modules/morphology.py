# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 09:50:01 2022

@author: MMBM_CAROLINE
"""
import cv2
import os
import numpy as np
import pandas as pd
import time
from skimage.feature import graycomatrix, graycoprops
import modules.functions as functions


def getArea(
        contours
    ):
    
    area = []
    
    for contour in contours:
        area.append(cv2.contourArea(contour))
        
    return area

def getCentroid(
        contours
    ):
    
    centroid = []
    
    for contour in contours:
        moment = cv2.moments(contour)
        
        if moment['m00'] != 0:
            cx = int(moment['m10']/moment['m00'])
            cy = int(moment['m01']/moment['m00'])
        
        else:
            cx = moment['m10']
            cy = moment['m10']
            
        centroid.append((cx,cy))
        
    return centroid

def getPerimeter(
        contours
    ):
    
    perimeter = []
    
    for contour in contours:
        perimeter.append(cv2.arcLength(contour,True))
        
    return perimeter

def getSolidity(
        contours
    ):
    
    solidity = []
    
    area = getArea(contours)
    
    for i in range(len(contours)):
        solidity.append(float(area[i])/cv2.contourArea(cv2.convexHull(contours[i])))
        
    return solidity
     
def getEquiDiameter(
        contours
    ):
    
    equi_diameter = []
    
    area = getArea(contours)
    for i in range(len(contours)):
        equi_diameter.append(np.sqrt(4*area[i]/np.pi))
        
    return equi_diameter

def getCirculartity(
        contours
    ):
    
    circularity = []
    
    area = getArea(contours)
    perimeter = getPerimeter(contours)
    
    for i in range(len(contours)):
        if perimeter[i]==0:
            circularity.append(1)
        else:
            circularity.append(4*np.pi*area[i]/perimeter[i]**2)
            
    return circularity
    
def getAspectRatio(
        contours
        ):
        
    aspect_ratio = []
    
    for contour in contours:
        if  contour.shape[0] > 5:
            # Fit contour on allipse
            ellipse = cv2.fitEllipse(contour)
            
            if max(ellipse[1]) !=0:
                # Compute mator and minor axis of ellipse
                major_axis = max(ellipse[1])
                minor_axis = min(ellipse[1])
                
                # Compute aspect ratio
                aspect_ratio.append(minor_axis/ major_axis)
                
            else:
                aspect_ratio.append(1)
        else: 
            aspect_ratio.append(1)
              
    return aspect_ratio

def getBackgroundMean(image, contour, kernelSizeOut, kernelSizeIn):
    "From an imade and a mask, compute average background value around the mask at a distance kernelSizeOut, don't take pixel very close to the mask "
    
    kernel_out = np.ones((kernelSizeOut, kernelSizeOut), np.uint8)
    kernel_in = np.ones((kernelSizeIn, kernelSizeIn), np.uint8)

    # Get mask of the spheroid
    mask = np.zeros(image.shape, np.uint8)
    mask = cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    dilated_mask_out = cv2.dilate(mask, kernel_out, iterations=1)
    dilated_mask_in = cv2.dilate(mask, kernel_in, iterations=1)
    mask_around_object = cv2.subtract(dilated_mask_out, dilated_mask_in)

    pixels_values = image[mask_around_object != 0]
    background = np.mean(pixels_values)

    return background, mask_around_object

def getMeanGreyLevel(
        image,
        contours, kernelSizeOut = 70, kernelSizeIn= 10, percentageKernelIn=1
    ):
    
    
    meanGreyLevels = []
    mask_greyLevel = []
    mask_background_greyLevel = []
    
    for i in range(len(contours)):
        background,mask_back = getBackgroundMean(image, contours[i], kernelSizeOut, kernelSizeIn)
        # Create a mask for the contour i
        mask = np.zeros(image.shape,np.uint8)
        mask = cv2.drawContours(mask,contours, i, 255, thickness=cv2.FILLED)
        
        _, _, w, h = cv2.boundingRect(contours[i])
        contour_size = max(w, h)
        # Calculate the kernel size
        kernel_size = int(contour_size * percentageKernelIn)
        kernel_size = 10
        # kernel_size = max(1, kernel_size // 2 * 2 + 1)
        # Create the kernel
        kernel_inin = np.ones((kernel_size, kernel_size), np.uint8)
        # Ensure the kernel size is odd and at least 1
        mask_erod = cv2.erode(mask,kernel_inin)
        # Get the pixel intensities inside the mask
        masked_pixels = np.where(mask_erod == 255, image, 0)
        pixel_values = masked_pixels[masked_pixels != 0]
        
        # Compute average pixel value 
        average_pixel_value = np.mean(pixel_values)
        
        normalized_greylevel = abs(average_pixel_value - background)

        meanGreyLevels.append(normalized_greylevel)
        mask_greyLevel.append(mask_erod)
        mask_background_greyLevel.append(mask_back)
    return meanGreyLevels, mask_greyLevel, mask_background_greyLevel

def getHomogeneity(image, contours):
    
    homogeneities = []
    energies = []
    correlations = []
    
    image = functions.convertTo8Bits(image)
    
    for contour in contours:
        # Create a mask for the contour
        mask = np.zeros_like(image, dtype=np.uint8)
        mask = cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Apply the mask to the image
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        # Calculate GLCM
        glcm = graycomatrix(masked_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        
        # Calculate homogeneity
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        
        homogeneities.append(homogeneity)
        energies.append(energy)
        correlations.append(correlation)
        
        
    return homogeneities, energies, correlations

def propertiesContours( # Compute all morphological properties of the contours and return them in a dataframe
        image, 
        contours
    ):
    masks = []
    
    for contour in contours:
        # Create a mask for the contour
        mask = np.zeros_like(image, dtype=np.uint8)
        mask = cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        masks.append(mask)
    
    area = getArea(contours)
    centroid = getCentroid(contours)
    perimeter = getPerimeter(contours)
    solidity = getSolidity(contours)
    equi_diameter = getEquiDiameter(contours)
    circularity = getCirculartity (contours)
    aspect_ratio = getAspectRatio(contours)
    meanGreyLevel, mask_greyLevel, mask_backgroundGreyLevel = getMeanGreyLevel(image, contours)
    homogeneity, energy, correlations = getHomogeneity(image, contours)

    properties = pd.DataFrame(list(zip(area,
                                       centroid,
                                       perimeter,
                                       solidity,
                                       equi_diameter,
                                       circularity,
                                       aspect_ratio,
                                       meanGreyLevel,
                                       homogeneity,
                                       energy,
                                       correlations
                                       )),
                              columns =['Area (pix2)', 
                                        'Centroid', 
                                        'Perimeter (pix)', 
                                        'Solidity', 
                                        'Equivalent Diameter (pix)', 
                                        'Circularity', 
                                        'Aspect ratio',
                                        'Mean grey value',
                                        'Homogeneity',
                                        'Energy',
                                        'Correlation'
                                        ])
    
    return properties,mask_greyLevel, mask_backgroundGreyLevel,  masks
    


