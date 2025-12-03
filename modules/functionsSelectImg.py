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


def _fill_internal_holes(mask, max_hole_area=None):
    """
    Fill internal holes in a binary mask (uint8, 0/255).
    Holes that touch the image border are left untouched. Optionally skip holes
    larger than `max_hole_area` (in pixels) when provided.
    Returns the mask with holes filled (in-place copy).
    """
    if mask is None or mask.size == 0:
        return mask
    h, w = mask.shape[:2]
    inv = cv2.bitwise_not(mask)
    # find contours of the inverted mask -> these are holes + external background
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # For each contour (candidate hole), fill it if it does NOT touch border
    mask_filled = mask.copy()
    for cnt in contours:
        if cnt is None or len(cnt) == 0:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        # if contour touches image border, skip (it's background)
        if x <= 0 or y <= 0 or (x + bw) >= (w - 1) or (y + bh) >= (h - 1):
            continue
        area = cv2.contourArea(cnt)
        if max_hole_area is not None and area > max_hole_area:
            continue
        cv2.drawContours(mask_filled, [cnt], -1, 255, -1)
    return mask_filled


def _fill_holes_until_border(mask, close_kernel_size=61, max_hole_area=None, force_convex=False):
    """
    Aggressively fill holes inside `mask` by first closing with a large kernel
    to bridge thin gaps, then filling all background connected components
    except the external background. This tends to remove internal holes
    completely (i.e. they become filled) and will also bridge narrow gaps
    that previously prevented filling.

    Parameters:
    - mask: uint8 binary mask (0/255)
    - close_kernel_size: int kernel diameter used for initial closing (odd)
    - max_hole_area: if set, skip filling holes larger than this area
    - force_convex: if True, compute convex hull of mask as final fallback
    """
    if mask is None or mask.size == 0:
        return mask
    # ensure binary 0/255
    mask_bin = (mask > 0).astype(np.uint8) * 255
    # Large closing to bridge narrow gaps
    try:
        ksz = close_kernel_size if close_kernel_size % 2 == 1 else close_kernel_size + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        mask_closed = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)
    except Exception:
        mask_closed = mask_bin.copy()

    # Inverted mask: background components
    inv = cv2.bitwise_not(mask_closed)
    # connected components on inverted mask (background regions)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inv, connectivity=8)
    # external background label is the label at (0,0)
    ext_label = labels[0, 0]
    mask_filled = mask_closed.copy()
    # fill all background components except external (i.e., internal holes)
    for lab in range(1, num_labels):
        if lab == ext_label:
            continue
        area = stats[lab, cv2.CC_STAT_AREA]
        if max_hole_area is not None and area > max_hole_area:
            continue
        mask_filled[labels == lab] = 255

    # optional convex hull fallback to ensure no remaining holes
    if force_convex:
        cnts, _ = cv2.findContours(mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_cnt = None
        if cnts:
            all_cnt = np.vstack(cnts)
        if all_cnt is not None and len(all_cnt) > 0:
            hull = cv2.convexHull(all_cnt)
            hull_mask = np.zeros_like(mask_filled)
            cv2.drawContours(hull_mask, [hull], -1, 255, -1)
            mask_filled = hull_mask

    # final smoothing
    kfinal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_filled = cv2.morphologyEx(mask_filled, cv2.MORPH_CLOSE, kfinal)
    return mask_filled


def _fill_holes_within_allowed(mask, allowed_mask=None, close_kernel_size=31, max_hole_area=None, min_overlap_frac=0.0):
    """
    Fill holes in `mask` but only where hole components overlap `allowed_mask`.
    This prevents filling outside the originally selected area while still
    filling internal holes that lie in the center of the allowed region.

    - mask: binary uint8 (0/255)
    - allowed_mask: binary uint8 (0/255) same size as mask; if None, behaves like aggressive fill
    - close_kernel_size: kernel size used to bridge narrow gaps before component fill
    - max_hole_area: skip holes larger than this value
    """
    if mask is None or mask.size == 0:
        return mask
    mask_bin = (mask > 0).astype(np.uint8) * 255
    try:
        ksz = close_kernel_size if close_kernel_size % 2 == 1 else close_kernel_size + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        mask_closed = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)
    except Exception:
        mask_closed = mask_bin.copy()

    inv = cv2.bitwise_not(mask_closed)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inv, connectivity=8)
    ext_label = labels[0, 0]
    mask_filled = mask_closed.copy()

    # Precompute allowed boolean if provided
    if allowed_mask is not None and allowed_mask.size == mask_filled.size:
        allowed_bool = (allowed_mask > 0)
    else:
        allowed_bool = None

    for lab in range(1, num_labels):
        if lab == ext_label:
            continue
        area = stats[lab, cv2.CC_STAT_AREA]
        if max_hole_area is not None and area > max_hole_area:
            continue
        comp = (labels == lab)
        # If allowed mask provided, only fill if overlap fraction exceeds threshold
        if allowed_bool is not None:
            overlap = np.count_nonzero(np.logical_and(comp, allowed_bool))
            hole_area = np.count_nonzero(comp)
            overlap_frac = float(overlap) / float(hole_area) if hole_area > 0 else 0.0
            if overlap_frac < float(min_overlap_frac):
                continue
        mask_filled[comp] = 255

    # final smoothing
    kfinal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask_filled = cv2.morphologyEx(mask_filled, cv2.MORPH_CLOSE, kfinal)
    return mask_filled


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


def detect_spheroids_in_zstack(
        paths,
        min_area=400,
        max_area=15000,
        min_circularity=0.2
    ):
    """
    Auto-detect spheroids from a z-stack of image paths.
    Returns a list of (cx, cy, area, circularity) for detected objects.
    Method: max projection, Gaussian blur, Otsu threshold, morphological clean, contour filter.
    """
    imgs = getImages(paths)
    if not imgs:
        return []

    # max projection to bring spheroid into focus
    stack = np.stack(imgs, axis=0)
    proj = np.max(stack, axis=0)

    # Convert to 8-bit for thresholding if needed (preserve relative intensities)
    if proj.dtype != np.uint8:
        maxv = proj.max() if proj.max() > 0 else 1
        proj8 = (proj.astype(np.float32) / float(maxv) * 255.0).clip(0, 255).astype(np.uint8)
    else:
        proj8 = proj

    # Smooth and threshold
    blur = cv2.GaussianBlur(proj8, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological clean
    kernel = np.ones((5, 5), np.uint8)
    clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim == 0:
            continue
        circularity = (4.0 * np.pi * area) / (perim * perim)
        if circularity < min_circularity:
            continue
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        detections.append((cx, cy, area, circularity))

    return detections


def detect_spheroids_blob(
        paths,
        min_sigma=5,
        max_sigma=50,
        num_sigma=10,
        threshold=0.02,
        overlap=0.5,
        projection='max'
    ):
    """
    Detect roughly circular spheroids using Laplacian of Gaussian blob detection.
    Returns list of (cx, cy, area, approx_circularity) tuples.
    """
    from skimage.feature import blob_log

    imgs = getImages(paths)
    if not imgs:
        return []

    # projection: 'max' or 'median'
    stack = np.stack(imgs, axis=0)
    if projection == 'median':
        proj = np.median(stack, axis=0)
    else:
        proj = np.max(stack, axis=0)

    # Convert to 2D grayscale (if color)
    if proj.ndim == 3:
        try:
            proj_gray = cv2.cvtColor(proj, cv2.COLOR_BGR2GRAY)
        except Exception:
            proj_gray = proj[..., 0]
    else:
        proj_gray = proj

    # Scale to float [0,1] for skimage
    proj_f = proj_gray.astype(np.float32)
    if proj_f.max() > 0:
        proj_f = proj_f / float(proj_f.max())

    # Run blob detection
    blobs = blob_log(proj_f, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold, overlap=overlap)
    # blob_log returns (y, x, sigma) where radius ~ sqrt(2)*sigma
    detections = []
    for y, x, sigma in blobs:
        radius = sigma * np.sqrt(2)
        area = np.pi * (radius ** 2)
        # approximate circularity as 1.0 for blob_log detections (can be refined)
        circularity = 1.0
        detections.append((int(x), int(y), float(area), float(circularity)))

    return detections


def detect_droplets_in_zstack(paths, min_area=20000, max_area=None, min_solidity=0.6, prefer_center=False, center_tolerance=0.25, min_aspect_for_center=2.0, max_gray_threshold=2200):
    """
    Detect droplet-shaped ovals from a z-stack. Returns a list of dicts with keys:
    - 'contour': contour points
    - 'mask': uint8 mask with droplet filled (same size as images)
    - 'centroid': (cx, cy)
    - 'bbox': (x,y,w,h)

    Uses max projection, blurring, Otsu threshold + morphological closing, then filters contours
    by area and solidity and fits ellipse when possible.
    """
    imgs = getImages(paths)
    if not imgs:
        return []

    stack = np.stack(imgs, axis=0)
    proj = np.max(stack, axis=0)

    # Ensure grayscale
    if proj.ndim == 3:
        try:
            proj_gray = cv2.cvtColor(proj, cv2.COLOR_BGR2GRAY)
        except Exception:
            proj_gray = proj[..., 0]
    else:
        proj_gray = proj

    # Remove very bright areas (likely reflections/edges) if threshold provided
    if max_gray_threshold is not None:
        try:
            bright_mask = proj_gray > max_gray_threshold
            proj_gray = proj_gray.copy()
            proj_gray[bright_mask] = 0
        except Exception:
            pass

    # Convert to 8-bit for morphology/thresholding
    if proj_gray.dtype != np.uint8:
        maxv = proj_gray.max() if proj_gray.max() > 0 else 1
        proj8 = (proj_gray.astype(np.float32) / float(maxv) * 255.0).clip(0, 255).astype(np.uint8)
    else:
        proj8 = proj_gray

    blur = cv2.GaussianBlur(proj8, (11, 11), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological closing to fill droplet interiors and remove thin lines
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = proj8.shape[:2]
    droplet_list = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area and area < min_area:
            continue
        if max_area and area > max_area:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull) if hull is not None else 0
        solidity = float(area) / hull_area if hull_area > 0 else 0
        if solidity < min_solidity:
            continue

        # Fit ellipse to check oval shape
        if cnt.shape[0] >= 5:
            try:
                ellipse = cv2.fitEllipse(cnt)
                (cx, cy), (MA, ma), angle = ellipse
                aspect = max(MA, ma) / (min(MA, ma) + 1e-6)
            except Exception:
                cx, cy = cv2.moments(cnt)['m10'] / (cv2.moments(cnt)['m00'] + 1e-6), cv2.moments(cnt)['m01'] / (cv2.moments(cnt)['m00'] + 1e-6)
                aspect = 1.0
        else:
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            aspect = 1.0

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        # keep a copy of the original mask (used to restrict hole-filling)
        mask_original = mask.copy()
        # Morphological closing to close larger gaps in contour and ensure filled interior
        try:
            # larger kernel to close wider gaps
            kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_fill)
            # small dilation to expand mask slightly and close narrow holes, then erode back
            kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask = cv2.dilate(mask, kd, iterations=2)
            mask = cv2.erode(mask, kd, iterations=2)
        except Exception:
            pass

        # Aggressively fill internal holes: close gaps and fill background components
        try:
            # restrict filling to the original mask area (dilated slightly)
            try:
                # slightly larger dilation to allow closing of narrow interior gaps
                allowed = cv2.dilate(mask_original, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=2)
            except Exception:
                allowed = mask_original
            # slightly larger close kernel and a slightly lower overlap requirement to close small remaining gaps
            mask = _fill_holes_within_allowed(mask, allowed_mask=allowed, close_kernel_size=31, max_hole_area=None, min_overlap_frac=0.03)
        except Exception:
            pass

        x, y, bw, bh = cv2.boundingRect(cnt)

        droplet_list.append({'contour': cnt, 'mask': mask, 'centroid': (int(cx), int(cy)), 'bbox': (x, y, bw, bh), 'area': area, 'solidity': solidity, 'aspect': aspect})

    # If prefer_center is True, try to pick the droplet that is elongated and near image center
    if prefer_center and len(droplet_list) > 0:
        h, w = proj8.shape[:2]
        img_cx = w / 2.0
        img_cy = h / 2.0
        best = None
        best_score = -1
        for d in droplet_list:
            (cx, cy) = d['centroid']
            x, y, bw, bh = d['bbox']
            aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
            # prefer wide/elongated, centered droplets
            center_dist = abs(cy - img_cy) / h
            width_frac = float(bw) / float(w)
            score = 0
            if aspect >= min_aspect_for_center:
                score += 2
            if center_dist <= center_tolerance:
                score += 2
            # wider bounding boxes get higher score
            score += width_frac
            if score > best_score:
                best_score = score
                best = d
        if best is not None:
            return [best]

    return droplet_list


def detect_spheroids_within_droplets(paths, droplet_list, method='blob', blob_params=None, inside=True, max_gray_threshold=2200):
    """
    For each droplet (dict as returned by detect_droplets_in_zstack), detect spheroids inside the droplet.
    Returns list of tuples (cx, cy, droplet_index).
    """
    from skimage.feature import blob_log

    if blob_params is None:
        blob_params = {'min_sigma':3, 'max_sigma':20, 'num_sigma':10, 'threshold':0.02, 'overlap':0.5}

    imgs = getImages(paths)
    if not imgs:
        return []

    stack = np.stack(imgs, axis=0)
    proj = np.max(stack, axis=0)
    if proj.ndim == 3:
        try:
            proj_gray = cv2.cvtColor(proj, cv2.COLOR_BGR2GRAY)
        except Exception:
            proj_gray = proj[..., 0]
    else:
        proj_gray = proj

    detections = []
    for i, droplet in enumerate(droplet_list):
        mask = droplet['mask']
        x, y, bw, bh = droplet['bbox']
        # Crop projection to droplet bbox for faster detection
        crop = proj_gray[y:y+bh, x:x+bw]
        mask_crop = mask[y:y+bh, x:x+bw]
        # Ensure per-droplet crop mask is closed and filled (stronger closing + flood fill)
        try:
                if mask_crop.size > 0:
                    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                    mask_crop_orig = mask_crop.copy()
                    mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_CLOSE, k2)
                    mask_crop = cv2.dilate(mask_crop, k2, iterations=2)
                    mask_crop = cv2.erode(mask_crop, k2, iterations=2)
                    # Aggressively fill holes but only inside a small dilation of the original crop mask
                    try:
                        # slightly larger allowed dilation on crop to help close narrow holes
                        allowed_crop = cv2.dilate(mask_crop_orig, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
                    except Exception:
                        allowed_crop = mask_crop_orig
                    # slightly lower overlap fraction to permit filling small interior gaps
                    mask_crop = _fill_holes_within_allowed(mask_crop, allowed_mask=allowed_crop, close_kernel_size=21, max_hole_area=None, min_overlap_frac=0.03)
                    # final smoothing
                    mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_CLOSE, k2)
        except Exception:
            pass

        # Prepare float projection scaled to 0..1
        crop_f = crop.astype(np.float32)
        maxv = float(crop_f.max()) if crop_f.max() > 0 else 1.0
        crop_f = crop_f / maxv

        # Apply mask: set outside to 0
        crop_f_masked = np.where(mask_crop == 255, crop_f, 0.0)

        # Mask out very bright pixels (in original crop) if requested
        if max_gray_threshold is not None:
            try:
                bright_crop_mask = crop > max_gray_threshold
                crop_f_masked = crop_f_masked.copy()
                crop_f_masked[bright_crop_mask] = 0.0
            except Exception:
                pass

        if method == 'blob':
            # If searching inside droplets, blob_input keeps interior; if searching outside, invert mask
            if inside:
                blob_input = crop_f_masked
            else:
                # outside region: keep values where mask_crop==0 and not too bright
                outside = np.where(mask_crop == 255, 0.0, crop_f)
                if max_gray_threshold is not None:
                    try:
                        outside[ crop > max_gray_threshold ] = 0.0
                    except Exception:
                        pass
                blob_input = outside
            blobs = blob_log(blob_input, min_sigma=blob_params['min_sigma'], max_sigma=blob_params['max_sigma'], num_sigma=blob_params['num_sigma'], threshold=blob_params['threshold'], overlap=blob_params['overlap'])
            for yb, xb, sigma in blobs:
                cx = int(x + xb)
                cy = int(y + yb)
                detections.append((cx, cy, i))
        else:
            # Fallback contour detection inside mask
            if inside:
                crop8 = (crop_f_masked * 255.0).clip(0,255).astype(np.uint8)
            else:
                outside8 = (np.where(mask_crop == 255, 0.0, crop_f) * 255.0)
                if max_gray_threshold is not None:
                    try:
                        outside8[ crop > max_gray_threshold ] = 0
                    except Exception:
                        pass
                crop8 = outside8.clip(0,255).astype(np.uint8)
            _, th = cv2.threshold(crop8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area < 50:
                    continue
                M = cv2.moments(cnt)
                if M['m00'] == 0:
                    continue
                cx_loc = int(M['m10']/M['m00'])
                cy_loc = int(M['m01']/M['m00'])
                detections.append((int(x+cx_loc), int(y+cy_loc), i))

    return detections

