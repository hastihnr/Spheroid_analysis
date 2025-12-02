# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 11:21:53 2023

@author: MMBM_CAROLINE
"""
import cv2
import os
import numpy as np
import pandas as pd
import time
import modules.functions as functions
import re


def getImages( # Return a list of images from a list of pathways
        paths, 
    ): 
    
    images = []
    
    for path in paths: 
        image = cv2.imread(path,-1)
        if type(image) != type(None) :
            images.append(image)
        
    return images

def removeStripeyInName(
        listFiles
    ):
    
    listRenamed = []
    
    for file in listFiles:
        if "STRIPEY" in file:
            listRenamed.append(file[8:])
        else: listRenamed.append(file)
        
    return listRenamed

def wellsNames(  # Generate list of all the well names possibilities
        lin, 
        col
    ): 
    
    allWells = []
   
    for l in range(1,lin+1):
        for c in range(1,col+1):
            if l<10:
                strL = "0"+str(l)
            else: strL = str(l)
            if c<10:
                strC = "0"+str(c)
            else: strC = str(c)
            
            allWells.append(strL+strC)
            
    return allWells

def getWellNamesOrdered(
        orientation
    ):
    # Old holder
    # c1 = [str(i)+"24" for i in range(16,9,-1)]+["0"+str(i)+"24" for i in range(9,0,-1)]
    # l1 = ["01"+str(i) for i in range(23,9,-1)]+["010"+str(i) for i in range(9,0,-1)]
    # c2 = ["0"+str(i)+"01" for i in range(2,10)]+[str(i)+"01" for i in range(10,16)]
    # l2 =  ["160"+str(i) for i in range(1,5)]
    # c3 = [str(i)+"05" for i in range(16,9,-1)]+["0"+str(i)+"05" for i in range(9,2,-1)]
    # l3 =  ["020"+str(i) for i in range(5,10)]+["02"+str(i) for i in range(10,23)]
    # c4 = ["0"+str(i)+"23" for i in range(2,10)]+[str(i)+"23" for i in range(10,17)]
    
    c1 = [str(i)+"24" for i in range(16,9,-1)]+["0"+str(i)+"24" for i in range(9,1,-1)]
    l1 = ["02"+str(i) for i in range(23,9,-1)]+["020"+str(i) for i in range(9,1,-1)]
    c2 = ["0"+str(i)+"01" for i in range(2,10)]+[str(i)+"01" for i in range(10,15)]
    l2 =  ["150"+str(i) for i in range(1,5)]
    c3 = [str(i)+"05" for i in range(15,9,-1)]+["0"+str(i)+"05" for i in range(9,3,-1)]
    l3 =  ["030"+str(i) for i in range(5,10)]+["03"+str(i) for i in range(10,22)]
    c4 = ["0"+str(i)+"22" for i in range(3,10)]+[str(i)+"22" for i in range(10,15)]
    
    wellsOrder = c1+l1+c2+l2+c3+l3+c4
    if orientation == 1 :
       wellsOrder.reverse() 

    return wellsOrder

def getWellNb(
        name,
        lin,
        col
    ):
    
    allWells = wellsNames(lin, col)
    
    pattern = r'\b\d{4}-\d{1,3}\b'
    
    if re.search(pattern, name):
        well = name[0:4]

    else:
        for i in allWells:
            if ('-'+i+'-') in name or ('-'+i+'.') in name:
                well = i
      
    return well

def plateFormat(
        well_plate_format
        ):
    
    if well_plate_format == 384: 
        lin = 16
        col = 24
        
    if well_plate_format == 96:
        lin = 8
        col = 12
    
    if well_plate_format == 24: 
        lin = 4
        col = 6
            
    if well_plate_format == 12: 
        lin = 3
        col = 4
        
    return (lin, col)

def getNImgStack(
        df_1well, 
        df_select,
        z
    ):
    
    # zs = int(z/2)
    df_infos_nz = pd.DataFrame(columns = df_1well.columns)
    
    df_1well = df_1well.sort_values(by=['NewName'])
    df_1well = df_1well.reset_index(drop=True)
    index_with_n = df_1well.index[df_1well['NewName'].isin(df_select['NewName'])].values[0]
    
    start_index = max(df_1well.index.min(), index_with_n - z)
    end_index = min(df_1well.index.max(), index_with_n + z)

    select_lines= df_1well.loc[start_index:end_index]
    print("nb images stacked =", len(select_lines))
    df_infos_nz = df_infos_nz.append(select_lines)

    return df_infos_nz

def getDataFrameInfosImg(
        folderPath,
        well_plate_format
    ): 
    
    (lin, col) = plateFormat(well_plate_format)
    
    # Get all .tiff names in the folder
    filesNames = [file for file in os.listdir(folderPath) if file.endswith("tiff")==True or file.endswith("tif")==True and "INVALID" not in file]
    
    # Remove "STIPEY" from the names
    filesNamesStripey = removeStripeyInName(filesNames)

    wells = [getWellNb(file, lin, col) for file in filesNames]
    
    paths = [folderPath+ "\\" + file for file in filesNames]

    #Get data frame with names, names wo STRIPEY, well number and image path
    df = pd.DataFrame(list(zip(filesNames,filesNamesStripey,wells,paths)), columns = ['Name', 'NewName', 'Wells', 'Paths'])
    
    return df


def getZstackImages(
        df,
        well):

    df_stack = df[df['Wells']==well]
    df_stack = df_stack.sort_values(by=['NewName'])
    paths_zstack = df_stack['Paths']
    
    return paths_zstack



def nameImages(spheros_liste):
        # Dictionnaire pour stocker l'indice courant pour chaque premier élément de tuple
    wells = {}
    # Liste pour stocker les résultats
    result = []
    # Compteur pour le changement de valeurs du premier élément du tuple
    i_counter = 1
    
    # Parcourir la liste de tuples
    for s in spheros_liste:
        well = s[0]
        
        if well not in wells:
            wells[well] = {'i': i_counter, 'ind': 0}
            i_counter += 1
        else:
            wells[well]['ind'] += 1
        
        # Construire la chaîne de caractères et l'ajouter à la liste des résultats
        i_value = wells[well]['i']
        ind_value = wells[well]['ind']
        result.append(f"{i_value}_{well}_{ind_value}")
    return result

