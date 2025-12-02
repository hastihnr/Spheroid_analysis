"""
This code is to select spheroids and it works for both brightfield and fluorescence images.
@author: Hasti Honari"""

import cv2
import os
import numpy as np
import pandas as pd
import time 
import modules.functions as functions
import modules.functionsSelectImg as functionsSelectImg
folder =  "E:\\Experiments\\PTT\\Spheroids\\08072025_hs578t\\1007 before laser"
#folder  = "H:\\Datas PE saved\\240617 Viability PDX time"



folderInputNames  = ['2025-07-10_102302_tube1']




well_plate_format = 384     #If droplet, 384, if microwells, 96

droplet = True  # True if droplet, false if wells

""""""
crop_size = 150  # pix
def scrolling(
        df,     # columns = ['Name', 'NewName', 'Wells', 'Paths'])
        well    
    ):
    """
    Displays a series of images for a specific well in a scrollable window and allows the user to select points/spheroids by 
    clicking on the images.

    Parameters:
    df (DataFrame): DataFrame containing image information. Expected columns are ['Name', 'NewName', 'Wells', 'Paths'].
    well (str): Identifier of the specific well to process.

    Returns:
    list: List of tuples [(well, cx,cy),...] with well identifiers and coordinates of selected points.
    """
    
    print("Well", well)
     
    # Get the image's paths corresponding to the specified  well (z-stack)
    paths_zstack = functionsSelectImg.getZstackImages(df, well)
    paths_zstack = paths_zstack.iloc[::4] # Only use every 4th image for faster processing

    # Load images
    images = functionsSelectImg.getImages(paths_zstack) 
    images = [functions.convertToRGB(i*16) for i in images] # Brighten images

    # Total number of image to display
    num_images = len(paths_zstack)
    
    # Create window to display the images
    cv2.namedWindow("Scrolling Images", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Scrolling Images", 480, 360)
    #cv2.resizeWindow("Scrolling Images", 1152, 864)
    # Variable for scrolling
    scroll_position = 1
    
    # List to store the well identifier and coordinates of selected points
    selected_sphero = []

    # Callback function for the trackbar to update the scroll position
    def on_trackbar(position):
        global scroll_position
        scroll_position = position
    
    # Callback function for mouse click events to select points on the image
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Draw a marker on the image at the clicked position
            #cv2.drawMarker(images[scroll_position], (x, y), color = [250, 0, 250],thickness = 3, markerType=cv2.MARKER_CROSS, markerSize=25)
            cv2.rectangle(images[scroll_position], (x - crop_size//2, y - crop_size//2), (x + crop_size//2, y + crop_size//2), color= [250, 0, 250], thickness = 3)
            # Save the well identifier and the coordinates of the click
            selected_sphero.append((well, x,y))
            print(f"Well {well} saved, coords {(x, y)}") 

    # Create a trackbar for scrolling through the images
    cv2.createTrackbar("Scroll", "Scrolling Images", int(num_images/3), num_images-1, on_trackbar)

    while True:
        # Get the current position of the scroll bar
        current_position = cv2.getTrackbarPos("Scroll", "Scrolling Images")
    
        # If the scroll position has changed, update the displayed image
        if current_position != scroll_position:
            scroll_position = current_position

        # Display image corresponding to the current scroll position
        cv2.imshow("Scrolling Images", images[scroll_position])
        
        # Set the callback function for mouse click events
        cv2.setMouseCallback('Scrolling Images', click_event)

        # Wait for a button to be pressed 10ms
        key = cv2.waitKey(10)
        
        # Leave the loop if ESC is pressed
        if key == 27: #ESC
            break

    # Close display window
    cv2.destroyAllWindows() 

    return selected_sphero

folderAnalysis = f"{folder}\\Analysis" 
functions.createFolder(folderAnalysis)

# Iterate through each input folder containing images
for f, folderInputName in enumerate(folderInputNames):
    t0 = time.time()
    print("Folder %s/%s : %s" %(f+1,len(folderInputNames), folderInputName))
    
    folderInputPath = f"{folder}\\{folderInputName}" 
    df_infos = functionsSelectImg.getDataFrameInfosImg(folderInputPath, well_plate_format)
    wells = df_infos["Wells"].drop_duplicates()
    
    if droplet:
        orderAllWells = functionsSelectImg.getWellNamesOrdered(1)
        wells = [w for w in wells if w in orderAllWells]
        wellsOrdered = sorted(wells, key=lambda x: orderAllWells.index(x))
    else: 
        wellsOrdered = wells.reset_index(drop=True).sort_values(ascending=True, ignore_index=False)
    
    cv2.destroyAllWindows()
    selected_spheros = []
    i = 0
    
    for well in wellsOrdered:
        selected_sphero = scrolling(df_infos, well)
        if selected_sphero:
            i += 1
            selected_spheros.extend(selected_sphero)
            print("Nb spheroids identified: ", i)
    
    print('Processing autofocus...')
    name_images = functionsSelectImg.nameImages(selected_spheros)
    
    folderImg8bCropPath = f"{folderAnalysis}\\{folderInputName}_cropped_8bits"
    functions.createFolder(folderImg8bCropPath)
    folderImg16bCropPath = f"{folderAnalysis}\\{folderInputName}_cropped_16bits"
    functions.createFolder(folderImg16bCropPath)
    folderCSVSelectPath = f"{folderAnalysis}\\selectedImagesCSV"
    functions.createFolder(folderCSVSelectPath)
    
    df_selected = pd.DataFrame()
    
    for ind, (well, cx, cy) in enumerate(selected_spheros):
        paths_zstack = functionsSelectImg.getZstackImages(df_infos, well).reset_index(drop=True)
        paths_BF = paths_zstack[paths_zstack.str.endswith("BF.tiff")].reset_index(drop=True)
        paths_flor = paths_zstack[~paths_zstack.str.endswith("BF.tiff")].reset_index(drop=True)
        #print(paths_flor)
        print(paths_BF)
        # print(paths_zstack[0])
        if not paths_BF.empty:
            images_BF = functionsSelectImg.getImages(paths_BF)
            images_cropped_BF = [functions.crop_image(image, cx, cy, crop_size) for image in images_BF]
            image_cropped_focused_BF, path_BF = functions.autoFocus(images_cropped_BF, paths_BF)
            image_cropped_focused_8bits_BF = functions.convertTo8Bits(image_cropped_focused_BF * 16)
            cv2.imwrite(f"{folderImg8bCropPath}\\{name_images[ind]}_8bits_BF.tiff", image_cropped_focused_8bits_BF)
            # Also save the original cropped focused image as 16-bit (no conversion)
            cv2.imwrite(f"{folderImg16bCropPath}\\{name_images[ind]}_16bits_BF.tiff", image_cropped_focused_BF.astype(np.uint16))
        
        if not paths_flor.empty:
            images_flor = functionsSelectImg.getImages(paths_flor)
            images_cropped_flor = [functions.crop_image(image, cx, cy, crop_size) for image in images_flor]
            image_cropped_focused_flor, path_flor = functions.autoFocus(images_cropped_flor, paths_flor)
            image_cropped_focused_8bits_flor = functions.convertTo8Bits(image_cropped_focused_flor * 16)
            image_cropped_focused_8bits_flor= cv2.convertScaleAbs(image_cropped_focused_8bits_flor, alpha=6, beta=3) #for very high signal
            # image_cropped_focused_8bits_flor= cv2.convertScaleAbs(image_cropped_focused_8bits_flor, alpha=18, beta=5) #for low signal
            #image_cropped_focused_8bits_flor= cv2.convertScaleAbs(image_cropped_focused_8bits_flor, alpha=255/image_cropped_focused_8bits_flor.max(), beta=1)
            cv2.imwrite(f"{folderImg8bCropPath}\\{name_images[ind]}_8bits_flor.tiff", image_cropped_focused_8bits_flor)
            print(f"{folderImg8bCropPath}\\{name_images[ind]}_8bits_flor.tiff")
            # Also save the original cropped focused fluorescence image as 16-bit (no conversion)
            cv2.imwrite(f"{folderImg16bCropPath}\\{name_images[ind]}_16bits_flor.tiff", image_cropped_focused_flor.astype(np.uint16))

        # Prepare the DataFrame row for the selected image: add the centroids of the detected object and an ID
        row_selected = df_infos[df_infos['Paths'] == path_BF].copy()
        row_selected['cx'] = cx
        row_selected['cy'] = cy
        row_selected['IDimage'] = name_images[ind]
        df_selected = df_selected.append(row_selected, ignore_index=True)

         # Save the DataFrame with selected images' information to a CSV file
    df_selected.to_csv(f"{folderCSVSelectPath}\\{folderInputName}_selectedImages.csv", index = False, float_format = str)
    
    print(f'Folder {f+1}/{len(folderInputNames)} done!')
    print("Processing time: %s sec " % round(time.time()-t0))
 
   