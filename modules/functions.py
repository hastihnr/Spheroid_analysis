# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 15:00:32 2022

@author: MMBM_CAROLINE
"""
import cv2
import os
import numpy as np
import pandas as pd
import time
import modules.functionsSelectImg as functionsSelectImg

def convertToRGB(
        imageConverted,
        normalize = 256
    ):
    
    bits = getImgBits(imageConverted)
    if bits == 16:
        imageConverted = (imageConverted/normalize).astype('uint8') # Move from 16bits to 8bits for debugging
    if len(imageConverted.shape) < 3:
        imageConverted = cv2.cvtColor(imageConverted, cv2.COLOR_GRAY2RGB) # Move from RGB so this allow color for debugging

    return imageConverted

def getImgBits(
        image
    ):
    
    if image.dtype == "uint16":
        bits = 16
    else:
        bits = 8
        
    return bits

def convertTo8Bits(
        image,
        normalize = 256
    ):
        bits = getImgBits(image)
        if bits != 8:
            image = (image/normalize).astype('uint8')
         
        return image

def show(
        windowName, # Name of the window
        image # Image to show
    ):

    screen_res = 1920, 1000
    scale_width = screen_res[0] / image.shape[1]
    scale_height = screen_res[1] / image.shape[0]
    scale = min(scale_width, scale_height)
   
    #resized window width and height
    window_width = int(image.shape[1] * scale)
    window_height = int(image.shape[0] * scale)

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowName, window_width, window_height)
    cv2.imshow(windowName, image)
    cv2.waitKey(1)
    
    return window_width, window_height

def saveImages(
        images_paths, 
        folder_path, 
        filenames, 
        brigth
    ):
    # Create folder
    createFolder(folder_path)
    
    # Save all the images contained in datafame
    for i in range(len(images_paths)):
        image = cv2.imread(images_paths[i],-1)
        if brigth == True: # Increase brighness to obtain non black img
            image = image*16
        cv2.imwrite(folder_path+'\\%s'%filenames[i],image)
        
        print("Done %s/%s" %(i+1,len(images_paths)))

def crop_image(
        image,
        center_x,
        center_y,
        crop_size
    ):
    
    half_size = crop_size // 2
    left = max(0, center_x - half_size)
    top = max(0, center_y - half_size)
    right = min(center_x + half_size, image.shape[1])
    bottom = min(center_y + half_size, image.shape[0])
    
    cropped_image = image[top:bottom, left:right]
    
    return cropped_image

    
    cropped_image = image[top:bottom, left:right]
    
    return cropped_image

def autoFocus(images, images_id):
    best_image = None
    best_sharpness = 0
    
    for i, image in enumerate(images): 
        
        sharpness = cv2.Laplacian (image, cv2.CV_64F).var()
        if sharpness > best_sharpness:
            best_sharpness = sharpness
            best_image = image
            best_image_id = images_id[i]

    return best_image, best_image_id

def merge_images(
        paths
    ):
    
    n = len(paths)
    
    z_stack_images = functionsSelectImg.getImages(paths)
    sum_image = np.sum(np.stack(z_stack_images), axis = 0)

    merged_image = (sum_image/n).astype(np.uint16)
    
    return merged_image

def createFolder(
        folder_path
    ):
    
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

def create_scale(
        size,
        scale #µm
        ):
    
    factor = 3.26 # 1pix = 3.26µm
    
    # Create white image
    image = np.ones((size, size, 3), dtype=np.uint8) * 255

    # Compute scale bar dimensions
    bar_lenght = int(scale / factor) 
    bar_width = int(scale / 20)

    # Draw scale bar on the center of the image
    x1 = (size - bar_lenght) // 2
    y1 = (size - bar_width) // 2
    x2 = x1 + bar_lenght
    y2 = y1 + bar_width
    image[y1:y2, x1:x2] = [0, 0, 0]  # Black rectangle

    # Ajouter le texte au centre de l'image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text = f"{scale} um"
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_position = ((size - text_size[0]) // 2, y2 + text_size[1] + 10)  # 10 pixels below scale bar
    cv2.putText(image, text, text_position, font, font_scale, [0, 0, 0], font_thickness, cv2.LINE_AA)

    return image
    

def getListFolders(path):
    # List of names
    folders = []
    
    # Parcourir tous les fichiers dans le dossier
    for elem in os.listdir(path):
        # Check if it's a folder
        if os.path.isdir(os.path.join(path, elem)):
            folders.append(elem)
        folders.sort(key=str.lower)         
    
    return folders