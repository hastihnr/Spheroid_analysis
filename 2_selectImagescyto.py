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
import morpho_functions as morpho_functions
# Disable interactive GUI by default during automated runs
SHOW_IMAGES = False
folder =  "E:\\Experiments\\PTT\\Spheroids\\08072025_hs578t\\1007 before laser"
#folder  = "H:\\Datas PE saved\\240617 Viability PDX time"



folderInputNames  = ['2025-07-10_102302_tube1']




well_plate_format = 384     #If droplet, 384, if microwells, 96

droplet = True  # True if droplet, false if wells

""""""
crop_size = 150  # pix
auto_mode = True  # set to True to run automatic spheroid detection instead of manual clicking
auto_method = 'blob'  # 'blob' or 'contour'
# search mode: when False, search outside detected droplets (i.e., omit droplet regions)
search_within_droplets = True
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
    if SHOW_IMAGES:
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
    if SHOW_IMAGES:
        cv2.createTrackbar("Scroll", "Scrolling Images", int(num_images/3), num_images-1, on_trackbar)

    while True:
        # Get the current position of the scroll bar
        current_position = cv2.getTrackbarPos("Scroll", "Scrolling Images")
    
        # If the scroll position has changed, update the displayed image
        if current_position != scroll_position:
            scroll_position = current_position

        if SHOW_IMAGES:
            # Display image corresponding to the current scroll position
            cv2.imshow("Scrolling Images", images[scroll_position])
            # Set the callback function for mouse click events
            cv2.setMouseCallback('Scrolling Images', click_event)
            # Wait for a button to be pressed 10ms
            key = cv2.waitKey(10)
            # Leave the loop if ESC is pressed
            if key == 27: #ESC
                break
        else:
            # Interactive display disabled; return empty selection
            print("Interactive display disabled (SHOW_IMAGES=False). Skipping manual selection.")
            break

    # Close display window
    if SHOW_IMAGES:
        cv2.destroyAllWindows() 

    return selected_sphero

folderAnalysis = f"{folder}\\Analysis" 

# Iterate through each input folder containing images
for f, folderInputName in enumerate(folderInputNames):
    t0 = time.time()
    print("Folder %s/%s : %s" %(f+1,len(folderInputNames), folderInputName))
    
    folderInputPath = f"{folder}\\{folderInputName}" 
    df_infos = functionsSelectImg.getDataFrameInfosImg(folderInputPath, well_plate_format)
    wells = df_infos["Wells"].drop_duplicates()
    functions.createFolder(folderAnalysis)
    folderImg8bCropPath = f"{folderAnalysis}\\{folderInputName}_cropped_8bits"
    functions.createFolder(folderImg8bCropPath)
    folderImg16bCropPath = f"{folderAnalysis}\\{folderInputName}_cropped_16bits"
    functions.createFolder(folderImg16bCropPath)
    folderCSVSelectPath = f"{folderAnalysis}\\selectedImagesCSV"
    functions.createFolder(folderCSVSelectPath)
    
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
        if auto_mode:
            # Automatic detection: use z-stack projection detector
            paths_zstack_full = functionsSelectImg.getZstackImages(df_infos, well).reset_index(drop=True)
            # First detect droplets (ovals) then detect spheroids inside them
            # Prefer the central elongated droplet (the thick line) for detection
            droplets = functionsSelectImg.detect_droplets_in_zstack(paths_zstack_full, prefer_center=True, center_tolerance=0.25, min_aspect_for_center=2.0)
            # optional: save QC projection with droplet outlines
            try:
                stack = functionsSelectImg.getImages(paths_zstack_full)
                proj = np.max(np.stack(stack, axis=0), axis=0)
                if proj.ndim == 3:
                    proj_vis = functions.convertToRGB(proj*16)
                else:
                    proj_vis = functions.convertToRGB(proj*16)
                for d in droplets:
                    cv2.drawContours(proj_vis, [d['contour']], -1, (0,255,0), 3)
                qc_folder = f"{folderAnalysis}\\{folderInputName}_detectionQC"
                functions.createFolder(qc_folder)
                cv2.imwrite(os.path.join(qc_folder, f"{well}_droplets_projection.png"), proj_vis)
                # Save individual droplet masks for QC
                for idx_d, d in enumerate(droplets):
                    mask_path = os.path.join(qc_folder, f"{well}_droplet_{idx_d}_mask.png")
                    # mask is uint8 single-channel; save directly
                    try:
                        cv2.imwrite(mask_path, d['mask'])
                    except Exception:
                        # fallback: convert to 3-channel for saving
                        try:
                            m3 = cv2.cvtColor(d['mask'], cv2.COLOR_GRAY2BGR)
                            cv2.imwrite(mask_path, m3)
                        except Exception:
                            pass
            except Exception:
                pass

            selected_sphero = []
            # If droplets were detected, run segmentation inside each droplet area using morpho_functions
            if droplets and len(droplets) > 0:
                # segmentation parameters (match 3_SegmentationAndMorpho defaults)
                min_aspect_ratio = 0
                max_aspect_ratio = 50.0
                min_circularity = 0.1
                max_circularity = 0.9
                min_area = 1000
                max_area = 2000
                gaussianFilterSD = 1
                threshold1 = 20
                threshold2 = 60

                for di, d in enumerate(droplets):
                    x, y, bw, bh = d['bbox']
                    # crop projection and mask for this droplet
                    try:
                        crop_proj = proj[y:y+bh, x:x+bw]
                        mask_crop = d['mask'][y:y+bh, x:x+bw]
                    except Exception:
                        continue

                    # prepare file paths for segmentation inputs in QC folder
                    seg_folder = qc_folder
                    os.makedirs(seg_folder, exist_ok=True)
                    proj16_path = os.path.join(folderImg16bCropPath, f"{well}_droplet_{di}_proj_16.tiff")
                    proj8_masked_path = os.path.join(folderImg8bCropPath, f"{well}_droplet_{di}_proj_masked_8.tiff")

                    # Save 16-bit projection crop if possible
                    try:
                        if crop_proj.dtype != np.uint16:
                            # try to preserve dynamic range by scaling
                            maxv = crop_proj.max() if crop_proj.max() > 0 else 1
                            crop16 = (crop_proj.astype(np.float32) / float(maxv) * 65535.0).clip(0, 65535).astype(np.uint16)
                        else:
                            crop16 = crop_proj
                        cv2.imwrite(proj16_path, crop16)
                    except Exception:
                        pass

                    # Save masked 8-bit projection used as contour image (mask out outside droplet)
                    try:
                        maxv = float(crop_proj.max()) if crop_proj.max() > 0 else 1.0
                        crop8 = (crop_proj.astype(np.float32) / maxv * 255.0).clip(0,255).astype(np.uint8)
                        crop8_masked = np.where(mask_crop == 255, crop8, 0).astype(np.uint8)
                        cv2.imwrite(proj8_masked_path, crop8_masked)
                    except Exception:
                        pass

                    # Run segmentation using morpho_functions; image_path uses 16-bit projection, contour path uses masked 8-bit
                    try:
                        seg_results = morpho_functions.process_image(proj16_path, proj8_masked_path, threshold1, threshold2, gaussianFilterSD, scaled_factor=0.8)
                        if seg_results:
                            # filter results
                            seg_filtered = morpho_functions.filter_objects(seg_results, min_aspect_ratio, max_aspect_ratio, min_circularity, max_circularity, min_area, max_area)
                            if seg_filtered:
                                # take first filtered contour as spheroid
                                contour_obj = seg_filtered[0]
                                cx_local = int(contour_obj.cx)
                                cy_local = int(contour_obj.cy)
                                cx_global = int(x + cx_local)
                                cy_global = int(y + cy_local)
                                selected_sphero.append((well, cx_global, cy_global))
                                # mark centroid on QC projection
                                try:
                                    cv2.circle(proj_vis, (cx_global, cy_global), 6, (0, 0, 255), -1)
                                except Exception:
                                    pass
                    except Exception:
                        pass

                # save projection with segmentation centroids
                try:
                    qc_detected_path = os.path.join(qc_folder, f"{well}_droplets_projection_detected.png")
                    cv2.imwrite(qc_detected_path, proj_vis)
                except Exception:
                    pass
            else:
                # If no droplets were detected, fallback to blob detection on entire projection
                dets_all = functionsSelectImg.detect_spheroids_blob(paths_zstack_full, min_sigma=3, max_sigma=30, threshold=0.02)
                selected_sphero = [(well, cx, cy) for (cx, cy, area, circ) in dets_all]
        else:
            selected_sphero = scrolling(df_infos, well)
        if selected_sphero:
            i += 1
            selected_spheros.extend(selected_sphero)
            print("Nb spheroids identified: ", i)
    
    print('Processing autofocus...')
    name_images = functionsSelectImg.nameImages(selected_spheros)
    
   
    
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
 
   