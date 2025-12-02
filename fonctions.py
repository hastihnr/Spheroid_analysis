import os
import cv2
import numpy as np
import pandas as pd
from skimage import feature
from skimage.util import img_as_ubyte 
import statistics
from skimage.feature import graycomatrix, graycoprops
gaussianFilterSD = 1
threshold1 = 0
threshold2 = 100  #65 for dark, 80 otherwise
class Contour:

    def __init__(self, contour, image_path, image, idx):
        self.contour = contour
        self.image_path = image_path
        self.image = image
        self.resized_contour = None
        self.idx = idx

    def resize_contour(self, scale_factor):
        # Get centroid of original contour
        M = cv2.moments(self.contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx = int(M['m10'] / 0.0001)
            cy = int(M['m01'] / 0.0001)

        # Resize contour
        resized_contour = self.contour * scale_factor

        # Translate resized contour to maintain the same center
        translated_contour = resized_contour + (cx * (1 - scale_factor), cy * (1 - scale_factor))

        # Append translated contour to the list
        self.resized_contour = translated_contour.astype(int)

    def get_original_contour(self):
        return self.contour

    def calculate_parameters(self, mean_gray_level, grey_in, grey_out):
        contour = self.resized_contour
        image = self.image
        # Calculate circularity
        self.circularity = calculate_circularity(contour)

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        self.area = calculate_area(contour)

        self.perimeter = calculate_perimeter(contour)

        self.cx, self.cy = calculate_centroid(contour)
        self.circularity = calculate_circularity(contour)
        self.aspect_ratio = calculate_aspectRatio(contour)
        #mean_gray_level =getMeanGreyLevel(image, contour)

        # Calculate mean gray level within the contour
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [contour], -1, (0,255,0), -1)

        # Calculate aspect ratio
        self.aspect_ratio = float(w) / h

        self.homogeneity = calculate_contour_homogeneity(image, contour)[0]
        self.energy = calculate_contour_homogeneity(image, contour)[1]
        self.correlation = calculate_contour_homogeneity(image, contour)[2]

        self.mean_gray_level = mean_gray_level
        self.grey_in = grey_in
        self.grey_out = grey_out

    def get_aspect_ratio(self):
        return self.aspect_ratio

    def get_circularity(self):
        return self.circularity
    
    def get_area(self):
        return self.area

    def get_resized_contour(self):
        return self.resized_contour

    def get_idx(self):
        return self.idx
    
    def get_image(self):
        return self.image

    def get_results(self):
        return self.idx, self.bckgrnd_img_path, self.area, self.perimeter, self.cx, self.cy, self.circularity, self.aspect_ratio, self.mean_gray_level, self.grey_in, self.grey_out, self.homogeneity, self.energy, self.correlation

    def set_background_image(self, image):
        self.bckgrnd_img = image
    
    def get_background_image(self):
        return self.bckgrnd_img
    
    def set_background_image_path(self, image_path):
        self.bckgrnd_img_path = image_path

    def get_background_image_path(self):
        return self.bckgrnd_img_path

def calculate_circularity(contour):
    if calculate_perimeter(contour) == 0:
        return 0
    return (4 * np.pi * calculate_area(contour)) / (calculate_perimeter(contour) ** 2)

 
def filter_objects(results, min_aspect_ratio, max_aspect_ratio, min_circularity, max_circularity, min_area, max_area):
    assert(len(results) > 0 and type(results[0]) is Contour), "Results must be a list of Contour objects" 
    filtered_results = []
    for contour in results:
        # Check if the object meets the filtering criteria
        if (min_aspect_ratio <= contour.get_aspect_ratio() <= max_aspect_ratio and
            min_circularity <= contour.get_circularity() <= max_circularity and
            min_area <= contour.get_area() <= max_area):
            filtered_results.append(contour)
    
    return filtered_results

    filtered_results = []
    
    for result in results:
        idx, image_path, area, perimeter, cx, cy, circularity, aspect_ratio, mean_gray_level, grey_in, grey_out, homogeneity, energy, correlation = result

        # Check if the object meets the filtering criteria
        if (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio and
            min_circularity <= circularity <= max_circularity and
            min_area <= area <= max_area):
            filtered_results.append(result)
            

    return filtered_results


def filteredContours(results,min_aspect_ratio, max_aspect_ratio, min_circularity, max_circularity, min_area, max_area,threshold1,threshold2,gaussianFilterSD):
    
    assert(len(results) > 0 and type(results[0]) is Contour), "Results must be a list of Contour objects"
    filResults = filter_objects(results,min_aspect_ratio, max_aspect_ratio, min_circularity, max_circularity, min_area, max_area)

    assert(len(filResults) <= 2), "There should be max one object after filtering"

    if len(filResults) == 0:
        return None, None

    contour_obj = filResults[0]
    idx = contour_obj.get_idx()
    contour = contour_obj.get_original_contour()

    image = contour_obj.get_background_image()
    image2 = convertToRGB(image, normalize = 256/16) # Bright image
    objects_img = cv2.drawContours(image2, [contour], -1, (0,255,0), 1)
#print(f"objects_img: {objects_img}")
    cv2.putText(objects_img,str(idx), calculate_centroid(contour),
    fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (255, 0, 255), thickness = 2)
    cv2.imshow("Image",objects_img)
    cv2.waitKey(0)
    return idx, objects_img

    filResults = filter_objects(results,min_aspect_ratio, max_aspect_ratio, min_circularity, max_circularity, min_area, max_area)

    if len(filResults) != 0:
        for i in filResults:
            image_path = i[1]
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            contours =  MaskMaker(image_path,threshold1,threshold2,gaussianFilterSD, scale_factor=1)
            index_contour = i[0]

            for idx, contour in enumerate(contours):
                if idx == index_contour:
                    image2 = convertToRGB(image, normalize = 256/16) # Bright image
                    objects_img = cv2.drawContours(image2, [contour], -1, (0,255,0), 1)
                #print(f"objects_img: {objects_img}")
                    cv2.putText(objects_img,str(idx), calculate_centroid(contour),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (255, 0, 255), thickness = 2)
                    cv2.imshow("Image",objects_img)
                    cv2.waitKey(0)
                    return idx, objects_img
    else:
        return None, "test"

def calculate_gray_level_difference(image, contour):
    normalised_grey = []
    # Create a mask for the contour
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Get the pixel intensities inside and outside the contour
    inside_pixels = image[mask == 255]
    outside_pixels = image[mask == 0]

    # Calculate the average gray level inside and outside the contour
    #mean_gray_inside = np.mean(inside_pixels)
    mean_gray_outside = np.mean(outside_pixels)

    for pixel in inside_pixels:
        pixel = pixel - mean_gray_outside
        pixel.append(normalised_grey)

    # Subtract the average gray level outside from the average gray level inside the contour
    #gray_level_difference = mean_gray_inside - mean_gray_outside

    return normalised_grey


def MaskMaker(image_path,threshold1,threshold2,gaussianFilterSD, scale_factor):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


    # Perform Canny edge detection
    edges = img_as_ubyte(feature.canny(image, gaussianFilterSD, threshold1, threshold2))
    cv2.imshow("edge",edges)
    #laplacian = cv2.Laplacian(image, -1, delta=100)
    #edges = np.uint8(np.absolute(laplacian))
    #edges = cv2.Laplacian(image, -1, ksize=5, scale=1,delta=5,borderType=cv2.BORDER_DEFAULT)
    #cv2.imshow("hi",edges)
    #cv2.waitKey(0)


    # Perform morphological operations to fill holes and close edges
    kernel = np.ones((5, 5), np.uint8)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    #print(edges_closed)
    cv2.imshow("edge",edges_closed)

    #cv2.imshow("hi",edges_closed)
    #cv2.waitKey(0)

    

    # Find contours
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_objects = []
    for contour in contours:
        contour_obj = Contour(contour, image_path, image, len(contours_objects))
        contour_obj.resize_contour(scale_factor)
        contours_objects.append(contour_obj)
    return contours_objects

    # previous implementation

    resized_contours = []
    # Resize contours
    resized_contours = []
    for contour in contours:
        # Get centroid of original contour
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx = int(M['m10'] / 0.0001)
            cy = int(M['m01'] / 0.0001)

        # Resize contour
        resized_contour = contour * scale_factor

        # Translate resized contour to maintain the same center
        translated_contour = resized_contour + (cx * (1 - scale_factor), cy * (1 - scale_factor))

        # Append translated contour to the list
        resized_contours.append(translated_contour.astype(int))

    return resized_contours
     # Scale down the contours
    #scaled_contours = []
    #for contour in contours:
        #print("shape is", np.shape(contour))
        #contour = contour*scale_factor
        #scaled_contours.append(contour_scaled)

    #return scaled_contours
    #return contours
def calculate_area(contour):
    area = cv2.contourArea(contour)
    return area
def calculate_perimeter(contour):
    perimeter = cv2.arcLength(contour, True)
    return perimeter

def calculate_centroid(contour):
    # Calculate centroid
        moment = cv2.moments(contour)
        if moment['m00'] != 0:
            cx = int(moment['m10'] / moment['m00'])
            cy = int(moment['m01'] / moment['m00'])
        else:
            cx = moment['m10']
            cy = moment['m10']
        return cx, cy

def calculate_bounding_rectangle(contour):
# Get bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    return x, y, w, h

def calculate_meanGrayLevel(image, contour):
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [contour], -1, (0,255,0), -1)
    mean_gray_level = cv2.mean(image, mask=mask)[0]
    return mean_gray_level

def calculate_aspectRatio(contour):
    x,y,w,h = calculate_bounding_rectangle(contour)
    aspect_ratio = float(w) / h
    
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
def getGreyOutside(image, unscaled_contours):

    for i in range(len(unscaled_contours)):
        # Create a mask for the contour i
        mask = np.zeros(image.shape,np.uint8)
        cv2.drawContours(mask,unscaled_contours,-1,(0,255,0), thickness=cv2.FILLED)
        outside_pixels = np.where(mask == 0, image,1)
        #print(len(outside_pixels))
    #     average_pixel_value = np.mean(masked_pixels)
        
    #     meanGreyLevels.append(average_pixel_value)
        
    # return meanGreyLevels

        outside_values = outside_pixels[outside_pixels!=1]
        mean_outside= np.mean(outside_values)

     
    return mean_outside

def getMeanGreyLevel(
        image,
        contours
    ):
    
    meanGreyLevels = []
    
    for i in range(len(contours)):
        # Create a mask for the contour i
        mask = np.zeros(image.shape,np.uint8)
        cv2.drawContours(mask,contours,-1,255, thickness=cv2.FILLED)
        # Get the pixel intensities inside the mask
        masked_pixels = np.where(mask == 255, image,0)

    #     average_pixel_value = np.mean(masked_pixels)
        
    #     meanGreyLevels.append(average_pixel_value)
        
    # return meanGreyLevels
        # outside_pixels = np.where(mask == 0, image,1)


        # outside_values = outside_pixels[outside_pixels!=1]
        # mean_outside= np.mean(outside_values)
    
        pixel_values = masked_pixels[masked_pixels != 0] 
        meanGreyLevels.append(pixel_values)
    #print(pixel_values)
        
        # Compute average pixel value 
        average_pixel_value = np.mean(meanGreyLevels)
        
        #meanGreyLevels.append(average_pixel_value)
        
    #return mean_outside, pixel_values, meanGreyLevels
    return average_pixel_value
def getImgBits(
        image
    ):
    
    if image.dtype == "uint16":
        bits = 16
    else:
        bits = 8
        
    return bits
# Function to calculate grey level homogeneity within a contour
def calculate_contour_homogeneity(image, contour):
    # Create a mask for the contour
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # Calculate GLCM
    glcm = graycomatrix(masked_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    # Calculate homogeneity
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    return homogeneity, energy, correlation

# Function to process each image
def process_image(image_path, contour_image_path, threshold1,threshold2,gaussianFilterSD, scaled_factor):

    # Read the image
    image = cv2.imread(image_path, -1) #cv2.IMREAD_GRAYSCALE
    #image = (image16/256).astype('uint8')

    # Calculate scaled and unscaled contours
    scaled_contours_objs =  MaskMaker(contour_image_path,threshold1,threshold2,gaussianFilterSD,scale_factor=scaled_factor)
    original_contours_objs =  MaskMaker(contour_image_path,threshold1,threshold2,gaussianFilterSD,scale_factor=1)

    scaled_contours = [contour.get_resized_contour() for contour in scaled_contours_objs]
    original_contours = [contour.get_resized_contour() for contour in original_contours_objs]

    # Calculate mean gray level within the contour
    mean_gray_level = abs(getMeanGreyLevel(image,scaled_contours) - getGreyOutside(image, original_contours))
    grey_in = getMeanGreyLevel(image,scaled_contours)
    grey_out = getGreyOutside(image, original_contours)
    print(getMeanGreyLevel(image,scaled_contours))

    for contour_obj in original_contours_objs:
        contour_obj.calculate_parameters(mean_gray_level, grey_in, grey_out)
        contour_obj.set_background_image(image)
        contour_obj.set_background_image_path(image_path)

    return original_contours_objs

    # Initialize results list
    results = []
    # Process each contour
    #for contour in contours:
    for idx, contour in enumerate(original_contours):



        # Calculate centroid
        # moment = cv2.moments(contour)
        # if moment['m00'] != 0:
        #     cx = int(moment['m10'] / moment['m00'])
        #     cy = int(moment['m01'] / moment['m00'])
        # else:
        #     cx = moment['m10']
        #     cy = moment['m10']

            contour = contour_obj.get_resized_contour()
            # Calculate circularity
            circularity = calculate_circularity(contour)
    
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            area = calculate_area(contour)

            perimeter = calculate_perimeter(contour)

            cx, cy = calculate_centroid(contour)
            circularity = calculate_circularity(contour)
            aspect_ratio = calculate_aspectRatio(contour)
            #mean_gray_level =getMeanGreyLevel(image, contour)

            # Calculate mean gray level within the contour
            mask = np.zeros_like(image)
            cv2.drawContours(mask, [contour], -1, (0,255,0), -1)

            
            #for pixel in getMeanGreyLevel(image, contour):
                #pixel = pixel - getGreyOutside(image, unscaled_contour)
                #pixels.append(pixel)

                 

            # Calculate aspect ratio
            aspect_ratio = float(w) / h

            homogeneity = calculate_contour_homogeneity(image, contour)[0]
            energy = calculate_contour_homogeneity(image, contour)[1]
            correlation = calculate_contour_homogeneity(image, contour)[2]

        
        #contour_index = np.where(contours == contour)

            # Append results to list
            
            results.append([idx,image_path,area, perimeter, cx, cy, circularity, aspect_ratio, mean_gray_level,grey_in,grey_out,homogeneity, energy, correlation])

    return results

# Function to process each image
def process_image_fluo(image_path, contour_image, threshold1,threshold2,gaussianFilterSD):
    
    image = cv2.imread(image_path, -1)
    #image_path_bf = image_path.replace('_flor', '_BF')

     #cv2.IMREAD_GRAYSCALE
    #image = (image16/256).astype('uint8')

    contours_objs =  MaskMaker(contour_image,threshold1,threshold2,gaussianFilterSD,scale_factor=1)
    unscaled_contours_objs =  MaskMaker(contour_image,threshold1,threshold2,gaussianFilterSD,scale_factor=1)

    contours = [contour.get_resized_contour() for contour in contours_objs]
    unscaled_contours = [contour.get_resized_contour() for contour in unscaled_contours_objs]


    mean_gray_level = getMeanGreyLevel(image,contours) - getGreyOutside(image, unscaled_contours)
    grey_in = getMeanGreyLevel(image,contours)
    grey_out = getGreyOutside(image, unscaled_contours)
    print(getMeanGreyLevel(image,contours))

    # Initialize results list
    results = []

    # Process each contour
    #for contour in contours:
    for idx, contour in enumerate(unscaled_contours):



        # Calculate centroid
        # moment = cv2.moments(contour)
        # if moment['m00'] != 0:
        #     cx = int(moment['m10'] / moment['m00'])
        #     cy = int(moment['m01'] / moment['m00'])
        # else:
        #     cx = moment['m10']
        #     cy = moment['m10']

            # Calculate circularity
            circularity = calculate_circularity(contour)
    
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            area = calculate_area(contour)

            perimeter = calculate_perimeter(contour)

            cx, cy = calculate_centroid(contour)
            circularity = calculate_circularity(contour)
            aspect_ratio = calculate_aspectRatio(contour)
            #mean_gray_level =getMeanGreyLevel(image, contour)

            # Calculate mean gray level within the contour
            mask = np.zeros_like(image)
            cv2.drawContours(mask, [contour], -1, (0,255,0), -1)

            
            #for pixel in getMeanGreyLevel(image, contour):
                #pixel = pixel - getGreyOutside(image, unscaled_contour)
                #pixels.append(pixel)

                 

            # Calculate aspect ratio
            aspect_ratio = float(w) / h

            homogeneity = calculate_contour_homogeneity(image, contour)[0]
            energy = calculate_contour_homogeneity(image, contour)[1]
            correlation = calculate_contour_homogeneity(image, contour)[2]

            



    
        

        
        #contour_index = np.where(contours == contour)

            # Append results to list
            results.append([idx,image_path,area, perimeter, cx, cy, circularity, aspect_ratio, mean_gray_level,grey_in,grey_out,homogeneity, energy, correlation])

    return results


def add_scale_bar(image, length_um, scale_factor_um_per_pixel, color=(0, 0, 0), thickness=2, text_color=(0, 0, 0)):
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Calculate the length of the scale bar in micrometers
    #length_um = length_pixels * scale_factor_um_per_pixel
    length_pixels = int(length_um/scale_factor_um_per_pixel)
    
    # Convert length to a human-readable format
    if length_um >= 1000:
        scale_text = f'{length_um / 1000:.1f} mm'
    else:
        scale_text = f'{length_um:.1f} um'
    
    # Calculate the length of the scale bar in pixels
    scale_length = int(length_pixels * (width / height))  # Adjust length based on image aspect ratio
    
    # Calculate the position of the scale bar
    scale_x = width - 20 - scale_length  # Position of the scale bar
    scale_y = height - 30  # Position of the scale bar
    
    # Draw the scale bar
    cv2.line(image, (scale_x, scale_y), (scale_x + scale_length, scale_y), color, thickness)
    
    # Add text indicating the scale
    cv2.putText(image, scale_text, (scale_x -10, scale_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, thickness=2)
    
    return image
def removeStripeyInName(
        listFiles
    ):
    
    listRenamed = []
    
    for file in listFiles:
        if "STRIPEY" in file:
            listRenamed.append(file[8:])
        else: listRenamed.append(file)
        
    return listRenamed