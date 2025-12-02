import os
import cv2
import numpy as np
import pandas as pd
from skimage import feature
from skimage.util import img_as_ubyte 
import fonctions as external_functions
from skimage.feature import graycomatrix, graycoprops
# Filter objects based on criteria
min_aspect_ratio = 0
max_aspect_ratio = 50.0
min_circularity = 0.1
max_circularity = 2.0
min_area = 400
max_area = 15000
gaussianFilterSD = 1
threshold1 = 30
threshold2 = 70  #65 for dark, 80 otherwise
threshold2_fluo = 100


# Path to the folder containing images

images_folder = "E:\\Experiments\\PTT\\Spheroids\\08072025_hs578t\\1007 before laser"
folderInputNames =  ['2025-07-10_102302_tube1']


for folderInputName in enumerate(folderInputNames):
    
    #print("Folder %s/%s : %s" %(f+1,len(folderInputNames), folderInputName))
    # Path to the folders where focused images will be found
    # 8-bit folder is used to create masks; 16-bit folder used for intensity measurements
    #input_folder = images_folder + "\\"+ "Morpho" + "\\%s_selectedstack" %folderInputName + "_Focused"
    input_folder_8 = os.path.join(images_folder, "Analysis", f"{folderInputName[1]}_cropped_8bits")
    input_folder_16 = os.path.join(images_folder, "Analysis", f"{folderInputName[1]}_cropped_16bits")
    contour_folder = os.path.join(images_folder, "Morpho", f"{folderInputName[1]}_Contours")
    os.makedirs(contour_folder, exist_ok=True)

    # List to store all results
    all_results = []
    filtered_results = []
    all_results_fluo = []
    filtered_results_fluo = []
 
    # Process each image in the 8-bit folder (masks) and map to corresponding 16-bit image for measurements
    for image_file in os.listdir(input_folder_8):
        if image_file.endswith(('.tiff', '.tif')):  # Filter only image files
            if image_file.endswith('_BF.tiff'):
                bf_image_path_8 = os.path.join(input_folder_8, image_file)
                # Map 8-bit filename to 16-bit counterpart (replace token `_8bits_` with `_16bits_`)
                if '_8bits_' in image_file:
                    image_file_16 = image_file.replace('_8bits_', '_16bits_')
                else:
                    # fallback: attempt replacing '8bits' with '16bits'
                    image_file_16 = image_file.replace('8bits', '16bits')
                bf_image_path_16 = os.path.join(input_folder_16, image_file_16)
                # If 16-bit version is missing, fall back to 8-bit (but warn)
                if not os.path.exists(bf_image_path_16):
                    print(f"[WARN] 16-bit image not found for '{image_file_16}'. Falling back to 8-bit for intensity calculations.")
                    bf_image_path_16 = bf_image_path_8

                # Use 8-bit image to create mask (contour_image_path) and 16-bit image (or fallback) for intensity measurements (image_path)
                results_objs = external_functions.process_image(bf_image_path_16, bf_image_path_8, threshold1, threshold2, gaussianFilterSD, scaled_factor=0.8)
                if not results_objs:
                    print(f"[INFO] No contours returned for '{image_file}', skipping.")
                    continue
                filtered_results_objs = external_functions.filter_objects(results_objs, min_aspect_ratio, max_aspect_ratio, min_circularity, max_circularity, min_area, max_area)
                filtered_results.extend([obj.get_results() for obj in filtered_results_objs])

            #print(filtered_results)
                all_results.extend(results_objs)
                contourFiltered = external_functions.filteredContours(results_objs,min_aspect_ratio, max_aspect_ratio, min_circularity, max_circularity, min_area, max_area,threshold1,threshold2,gaussianFilterSD)
                print(contourFiltered)
                if contourFiltered[0] != None:

                 # Save the image with filtered contour
                    if not os.path.exists(contour_folder):
                        os.makedirs(contour_folder)
                    filename = os.path.splitext(image_file)[0]  # Extract filename without extension
                    new_filename = f"{filename}_{contourFiltered[0]}"  # Combine filename with contourFiltered[0]
                    contour_file = os.path.join(contour_folder, f"{new_filename}.tiff")
                    with_scale = external_functions.add_scale_bar(contourFiltered[1], 200, 3.26,color=(0, 0, 0), thickness=3, text_color=(0, 0, 0))
                #print(with_scale)
                    cv2.imwrite(contour_file, with_scale)
                    flor_image_file = image_file.replace("BF.tiff", "flor.tiff")
                    flor_image_path_8 = os.path.join(input_folder_8, flor_image_file)
                    # Map 8-bit flor filename to 16-bit counterpart
                    if '_8bits_' in flor_image_file:
                        flor_image_file_16 = flor_image_file.replace('_8bits_', '_16bits_')
                    else:
                        flor_image_file_16 = flor_image_file.replace('8bits', '16bits')
                    flor_image_path_16 = os.path.join(input_folder_16, flor_image_file_16)
                    if os.path.exists(flor_image_path_16):
                        # Use BF-derived mask (8-bit) and fluorescence 16-bit image for measurements
                        flor_results = external_functions.process_image(flor_image_path_16, bf_image_path_8, threshold1, threshold2, gaussianFilterSD, scaled_factor=1)
                        if not flor_results:
                            print(f"[INFO] No contours returned for fluorescence '{flor_image_file}', skipping.")
                        else:
                            flor_filtered_results = external_functions.filter_objects(flor_results, min_aspect_ratio, max_aspect_ratio, min_circularity, max_circularity, min_area, max_area)
                        flor_all_results = []
                        flor_all_results.extend(flor_results)
                        contourFiltered = external_functions.filteredContours(flor_results, min_aspect_ratio, max_aspect_ratio, min_circularity, max_circularity, min_area, max_area, threshold1, threshold2, gaussianFilterSD)
                        filtered_results_fluo_objs = external_functions.filter_objects(flor_results, min_aspect_ratio, max_aspect_ratio, min_circularity, max_circularity, min_area, max_area)
                        filtered_results_fluo.extend([obj.get_results() for obj in filtered_results_fluo_objs])

                    # if flor_contourFiltered[0] is not None:
                    #     # Save the image with filtered contour for -flor.tiff
                    #     flor_new_filename = f"{filename}_flor_{flor_contourFiltered[0]}"
                    #     flor_contour_file = os.path.join(contour_folder, f"{flor_new_filename}.tiff")
                    #     flor_with_scale = f.add_scale_bar(flor_contourFiltered[1], 200, 3.26, color=(0, 0, 0), thickness=3, text_color=(0, 0, 0))
                    #     cv2.imwrite(flor_contour_file, flor_with_scale)
            else:
                print(f"No contours found for {image_file}")
            # else:
            #     image_path = os.path.join(input_folder, image_file)
            #     contour_img = os.path.join(input_folder, image_file)
            #     contour_img = contour_img.replace('_flor', '_BF')
            #     print(contour_img)
            #     results_fluo = f.process_image_fluo(image_path,contour_img, threshold1,threshold2,gaussianFilterSD)
            #     filtered_results_fluo.extend(f.filter_objects(results_fluo, min_aspect_ratio, max_aspect_ratio, min_circularity, max_circularity, min_area, max_area))
            # #print(filtered_results)
            #     all_results_fluo.extend(results_fluo)
            #     contourFiltered_fluo = f.filteredContours(results_fluo,min_aspect_ratio, max_aspect_ratio, min_circularity, max_circularity, min_area, max_area,threshold1,threshold2_fluo,gaussianFilterSD)
            #     # print(
            #     #     f"Filtered results: {contourFiltered_fluo}"
            #     # )
            
            #     if contourFiltered_fluo is not None and contourFiltered_fluo[0] is not None:

            #      # Save the image with filtered contour
            #         if not os.path.exists(contour_folder):
            #             os.makedirs(contour_folder)
            #         filename = os.path.splitext(image_file)[0]  # Extract filename without extension
            #         new_filename = f"{filename}_{contourFiltered_fluo[0]}"  # Combine filename with contourFiltered[0]
            #         contour_file = os.path.join(contour_folder, f"{new_filename}.tiff")
            #         with_scale = f.add_scale_bar(contourFiltered_fluo[1], 200, 3.26,color=(0, 0, 0), thickness=3, text_color=(0, 0, 0))
            #     #print(with_scale)
            #         cv2.imwrite(contour_file, with_scale)
            #     else:
            #         pass


    # Create DataFrame from filtered results
    df = pd.DataFrame(filtered_results, columns=["Index","File","Area", "Perimeter", "Centroid_X", "Centroid_Y", "Circularity", "Aspect_Ratio", "Mean_Gray_Level", "Grey_in", "Grey_out", "Homogeneity", "Energy", "Correlation"])
    df_fluo = pd.DataFrame(filtered_results_fluo, columns=["Index","File","Area", "Perimeter", "Centroid_X", "Centroid_Y", "Circularity", "Aspect_Ratio", "Mean_Gray_Level", "Grey_in", "Grey_out", "Homogeneity", "Energy", "Correlation"])
    #df = df.transpose()
    df.to_csv(os.path.join(contour_folder, "Morpho.csv"), index=False, sep=';')
    df_fluo.to_csv(os.path.join(contour_folder, "Morpho_fluo.csv"), index=False, sep=';')

    print("DataFrame created with filtered results:")
    print(df)
