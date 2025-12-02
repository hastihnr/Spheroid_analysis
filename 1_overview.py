# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 16:36:11 2023

@author: MMBM_CAROLINE
"""
import cv2
import os
import numpy as np
import pandas as pd
import time
import modules.functions as functions
import modules.functionsSelectImg as functionsSelectImg

"""
This script is used to create an "overview" of the images taken with the plate reader. 
Theoverviews images are created for each folder of image given in folderInputNames, and stored in a new folder overviews.

folderInputNames: names of the folders containing the images. 
folder: path of the parent folder that contains the folders of images 
well_plate_format: plate format imaged
z_nb: focal plan displayed
scale_percent: percentage of the initial images displayed (if 100, no size reduction, but heavy file at the end)

"""

""""""
folder =  "F:\\Experiments\\PTT\\Spheroids\\05052025\\07052025 after laser"
#folder  = "H:\\Datas PE saved\\240617 Viability PDX time"


# folderInputNames = ['2024-01-30_141717_tube1_0um_day1',
#                      '2024-01-30_142935_tube2_0.1um_day1',
#                      '2024-01-30_144217_tube3_1um_day1',
#                      '2024-01-30_145531_tube4_2um_day1',
#                      '2024-01-30_150814_tube5_5uM_day1',
#                      '2024-01-30_151910_tube6_10uM_day1',
#                      '2024-01-30_153121_tube7_20uM_day1',
#                      '2024-01-30_154642_tube8_50uM_day1',
#                      '2024-02-01_131519_tube1_0um_day3',
#                      '2024-02-01_132641_tube2_0.1um_day3',
#                      '2024-02-01_133802_tube3_1um_day3',
#                      '2024-02-01_134954_tube4_2um_day3',
#                      '2024-02-01_140132_tube5_5uM_day3',
#                      '2024-02-01_141330_tube6_10uM_day3',
#                      '2024-02-01_142451_tube7_20uM_day3',
#                      '2024-02-01_143732_tube8_50uM_day3']
folderInputNames  = ['2025-05-07_135107_tube1','2025-05-07_140511_tube3','2025-05-07_141932_tube7','2025-05-07_143332_tube4','2025-05-07_145019_tube2',
'2025-05-07_150413_tube5','2025-05-07_151807_tube8','2025-05-07_153210_tube6']


well_plate_format = 384  # If droplet, 384, if microwells, 96

""""""

z_nb = [9] # Stack desired
scale_percent = 30 # Percentage of the initial image size
crop_x, crop_y = (460, 50)  # (460, 50) for well_plate_format = 384    (0, 0) for well_plate_format = 96

def overview( # Create and return an overview of the plate with image resized, from a data frame containing images infos
        df,                  # dataframe containing the informations of the images to plot, columns = ['Name', 'NewName', 'Wells', 'Paths'])
        scale_percent,       # Percentage of the initial image size
        nbcol,
        nblin,
        crop_x,         # pix   # 460 for well_plate_format = 384
        crop_y,         # pix   #  50 for well_plate_format = 384
        width_img_ini = 1920,    # pix  size of the images taken by the plate reader
        height_img_ini = 1440    # pix

    ):
    """
    Creates and returns an overview of the plate with images resized and arranged according to the data frame containing image information.

    Parameters:
    df (DataFrame): DataFrame containing the information of the images to plot. Expected columns are ['Name', 'NewName', 'Wells', 'Paths'].
    scale_percent (int): Percentage of the initial image size for resizing.
    nbcol (int): Number of columns in the overview image.
    nblin (int): Number of lines in the overview image.
    crop_x (int): Pixel value for cropping width (e.g., 460 for well_plate_format = 384).
    crop_y (int): Pixel value for cropping height (e.g., 50 for well_plate_format = 384).
    width_img_ini (int): Initial width of the images taken by the plate reader. Default is 1920 pixels.
    height_img_ini (int): Initial height of the images taken by the plate reader. Default is 1440 pixels.

    Returns:
    np.ndarray: The overview image as a NumPy array.
    """
    
    # Compute images size after cropping and size reduction
    width_img = int((width_img_ini - crop_y)* scale_percent / 100)
    height_img = int((width_img_ini - crop_x) * scale_percent / 100)
    
    # Define overview image size by multiplying the size of each image by the number of lines/columns
    width_over = width_img * nbcol
    height_over = height_img * nblin

    # Initialize overview image, black image 8 bits
    img_overview = np.zeros((height_over, width_over), dtype=np.uint8) 
    
    # Find the position (line and column) of each image of the dataframe, add 'Lines' and 'Columns' in the dataframe
    lines = [int(w[0:2]) for w in df['Wells']]
    cols = [int(w[2:4]) for w in df['Wells']]
    df['Lines'] = lines
    df['Columns'] = cols
    
    # Print each image at its position in the plate on the overview image
    for i in range(len(df)):
        # Get position and path of the image
        path = df['Paths'][i]
        lin = df['Lines'][i]
        col = df['Columns'][i]
        
        # Load the image
        image = cv2.imread(path, -1)
        
        # Crop and resize the image
        image = image[(crop_y//2):height_img_ini-(crop_y//2), (crop_x//2):width_img_ini-(crop_x//2)]
        image = cv2.resize(image, (width_img,height_img))
        
        # Bright the image and convert to 8bits
        image = image*16
        image = cv2.convertScaleAbs(image, alpha=(255.0/65535.0))
        
        # Get the position of the image (up left corner cordinates)
        x_pos = (col - 1) * width_img
        y_pos = (lin - 1) * height_img
        
        # Display the image at it's position on the overview
        img_overview[y_pos:y_pos+height_img, x_pos:x_pos+width_img] = image
        
    return img_overview


""""""
folderAnalysis = f"{folder}\\Analysis" 
functions.createFolder(folderAnalysis)
(nblin, nbcol) = functionsSelectImg.plateFormat(well_plate_format)

for f, folderInputName in enumerate(folderInputNames):
    print("Folder %s/%s : %s" %(f+1,len(folderInputNames), folderInputName))
    
    # Create a folder that will contain the overviews images
    folderInputPath = f"{folder}\\{folderInputName}"
    folderOverviewPath = f"{folderAnalysis}\\overviews"      
    functions.createFolder(folderOverviewPath)

    # Data frame with all the file names & infos  ; columns = ['Name', 'NewName', 'Wells', 'Paths'])
    df_infos = functionsSelectImg.getDataFrameInfosImg(folderInputPath, well_plate_format)

    # Get wells names present in the folder and the nb of z images per well
    wells = df_infos["Wells"].drop_duplicates()
    
    # Create & save an overview image for each z 
    for i, z in enumerate(z_nb):
        # Create a dataframe with the images' infos corresponding to the same z
        df_overview = pd.DataFrame(columns = df_infos.columns)
        for well in wells:
            df_1well = df_infos[df_infos['Wells']==well]
            df_1well = df_1well.sort_values(by=['NewName'])
            df_overview = df_overview.append(df_1well.iloc[z], ignore_index = True)
        
        # Create imaage overview for the same z
        img_over = overview(df_overview, scale_percent, nbcol, nblin, crop_x, crop_y)
        
        # Save image overview
        cv2.imwrite(f"{folderOverviewPath}\\{folderInputName}_{z}.tiff", img_over)
        
        #cv2.imshow("img_over", img_over)     
        
        print("Overview %s/%s" %(i+1,len(z_nb)))
