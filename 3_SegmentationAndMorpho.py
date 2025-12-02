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

images_folder = "F:\\Experiments\\PTT\\Spheroids\\05052025\\07052025 after laser"
folderInputNames =  ['2025-05-07_135107_tube1','2025-05-07_140511_tube3','2025-05-07_141932_tube7','2025-05-07_143332_tube4','2025-05-07_145019_tube2',
'2025-05-07_150413_tube5','2025-05-07_151807_tube8','2025-05-07_153210_tube6']


for folderInputName in enumerate(folderInputNames):
    
    #print("Folder %s/%s : %s" %(f+1,len(folderInputNames), folderInputName))
    # Path to the folder where focused images will be found
    #input_folder = images_folder + "\\"+ "Morpho" + "\\%s_selectedstack" %folderInputName + "_Focused"
    input_folder = os.path.join(images_folder, "Analysis", f"{folderInputName[1]}_cropped_8bits")
    contour_folder = os.path.join(images_folder, "Morpho", f"{folderInputName[1]}_Contours")
    os.makedirs(contour_folder, exist_ok=True)

    # List to store all results
    all_results = []
    filtered_results = []
    all_results_fluo = []
    filtered_results_fluo = []
 
    # Process each image in the folder
    for image_file in os.listdir(input_folder):
        if image_file.endswith(('.tiff', '.tif')):  # Filter only image files
            if image_file.endswith('_BF.tiff'):
                bf_image_path = os.path.join(input_folder, image_file)
                results_objs = external_functions.process_image(bf_image_path, bf_image_path,threshold1,threshold2,gaussianFilterSD, scaled_factor=0.8)
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
                    flor_image_path = os.path.join(input_folder, flor_image_file)
                    if os.path.exists(flor_image_path):
                        flor_results = external_functions.process_image(flor_image_path, bf_image_path, threshold1, threshold2, gaussianFilterSD, scaled_factor=1)
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
