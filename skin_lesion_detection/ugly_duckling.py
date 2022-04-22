from audioop import mul
import os
import cv2
import sys
import math
import pandas as pd
import time
import pickle
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import  MinMaxScaler
from tensorflow.keras import applications
from keras.preprocessing.image import img_to_array
from keras.models import load_model, Model
from keras.applications.imagenet_utils import preprocess_input

img_width, img_height = 150, 150 # for VGG model

#Masking
def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

#Threshold binarization
def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)
    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)
    return matrix

# Simple color balance algorithm using Python 3.7 and OpenCV 
#       (Based on code from DavidYKay https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc#file-simple_cb-py) 
def color_balance(img, percent):
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 200.0
    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        # find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1
        flat = np.sort(flat)
        n_cols = flat.shape[0]

        low_val  = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil( n_cols * (1.0 - half_percent))]

        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

def apply_clahe(image, c_lim=1.0):
    #-----Converting image to LAB Color model----------------------------------- 
    lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    #-----Applying CLAHE to L-channel-------------------------------------------
    tile_L = int(math.sqrt(np.shape(image)[0]*np.shape(image)[1])/100)
    if tile_L<1: tile_L=1
    clahe = cv2.createCLAHE(clipLimit=c_lim, tileGridSize=(tile_L,tile_L))
    cl = clahe.apply(l)
    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))
    #-----Converting image from LAB Color model to RGB model--------------------
    image_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return image_clahe

def skin_detector(image):
    ## Inspired by naive skin detectors from:
    #    https://github.com/Jeanvit/PySkinDetection
    #    https://github.com/CHEREF-Mehdi/SkinDetection
    
    #Converting from gbr to hsv color space
    img_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #skin color range for hsv color space 
    HSV_mask_pos = cv2.inRange(img_HSV, (0, 3, 0), (35,255,255)) 
    HSV_mask_neg = cv2.inRange(img_HSV, (154, 3, 0), (179,255,255))
    HSV_mask=cv2.bitwise_or(HSV_mask_pos,HSV_mask_neg)
    #HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    
    #converting from gbr to YCbCr color space
    img_YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    #skin color range for hsv color space 
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 130, 77), (255,180,130)) 
    #cv2.imshow("YCrCbWindow", YCrCb_mask)
    #YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    
    #merge skin detection (YCbCr and hsv)
    global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
    global_mask=cv2.medianBlur(global_mask,3)
    #global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    
    return global_mask

# Define multiscale mole id function using OPENCV's simple blob detection module
def get_multiscale_moles(image, CLAHE_Adj = False):
    
    # Grayscale convertion
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (im_height,im_width) =img_gray.shape[:2]
    
    # create a CLAHE object (Arguments are optional).
    if CLAHE_Adj==True:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        im = clahe.apply(img_gray)
    else:
        im = img_gray 
    
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    
    # Filter by thresholds
    params.minThreshold = 0
    params.maxThreshold = 255
    
    # Filter by Area.
    params.filterByArea = True
    params.minArea = (10*10) #10x10 Pixel limit for analysis
    params.maxArea = (im_height*im_width)
    
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1
    
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.1
    
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.1
    
    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        detector = cv2.SimpleBlobDetector(params) #Command for Python 2.7
    else : 
        detector = cv2.SimpleBlobDetector_create(params) #Command for Python 3.5
        
    keyPoints = detector.detect(im)
    n_blobs = len(keyPoints)
    ROI_blobs = np.zeros((n_blobs,3),np.uint64)
    #i is the index of the blob you want to get the position
    i=0
    for keyPoint in keyPoints:
        ROI_blobs[i,0] = keyPoint.pt[0]  #Blob X coordinate
        ROI_blobs[i,1] = keyPoint.pt[1]  #Blob Y coordinate
        ROI_blobs[i,2] = keyPoint.size   #Blob diameter (average)
        i+=1
    
    # Draw detected blobs as red circles.
    # Note that cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keyPoints = cv2.drawKeypoints(im, keyPoints, np.array([]), (255,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return ROI_blobs, n_blobs, im_with_keyPoints

# Define mole center locator function using OPENCV's simple blob detection module
def get_center_mole(image, CLAHE_Adj = False):
    
    # Grayscale convertion
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (im_height,im_width) =img_gray.shape[:2]
    
    # create a CLAHE object (Arguments are optional).
    if CLAHE_Adj==True:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        im = clahe.apply(img_gray)
    else:
        im = img_gray 
    
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Filter by thresholds
    params.filterByColor = True
    params.blobColor = 0
    params.minThreshold = 0
    params.maxThreshold = 255
    
    # Filter by Area.
    params.filterByArea = True
    params.minArea = (im_height*im_width)*33/1000 # Pixel limit for analysis
    params.maxArea = (im_height*im_width)*660/1000
    
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1
    
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.1
    
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.1
    
    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        detector = cv2.SimpleBlobDetector(params) #Command for Python 2.7
    else : 
        detector = cv2.SimpleBlobDetector_create(params) #Command for Python 3.5
        
    keyPoints = detector.detect(im)
    n_blobs = len(keyPoints)
    ROI_blobs = np.zeros((n_blobs,3),np.uint64)
    #i is the index of the blob you want to get the position
    i=0
    for keyPoint in keyPoints:
        ROI_blobs[i,0] = keyPoint.pt[0]  #Blob X coordinate
        ROI_blobs[i,1] = keyPoint.pt[1]  #Blob Y coordinate
        ROI_blobs[i,2] = keyPoint.size   #Blob diameter (average)
        i+=1
    
    # Draw detected blobs as red circles.
    # Note that cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keyPoints = cv2.drawKeypoints(im, keyPoints, np.array([]), (255,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return ROI_blobs, n_blobs, im_with_keyPoints

### Define multiscale spl id and classification function using OPENCV's  blob detection and CNN classifier
def multiscale_wide_field_spl_extraction(wf_orig_image, model, im_dim=[img_width,img_height], layer_name = 'dense', display_plots=False):
    
    # Specify Display window settings
    if display_plots==True:
        cv2.namedWindow("SPLWindow", cv2.WINDOW_NORMAL)        # Suspicious Pigmented Lesion Tracking window
        cv2.moveWindow("SPLWindow", 0,20)
        
        cv2.namedWindow("MoleWindow", cv2.WINDOW_NORMAL)       # Mole detection window
        cv2.moveWindow("MoleWindow", 0,360)
        
        cv2.namedWindow("SLAWindow", cv2.WINDOW_NORMAL)        # Single-Lesion Analysis Window
        cv2.moveWindow("SLAWindow", 0,695)
        
        cv2.namedWindow("CAMWindow", cv2.WINDOW_NORMAL)        # Convolutional Activation Map (single-lesion crop)
        cv2.moveWindow("CAMWindow", 405,20)

        cv2.namedWindow("MASWindow", cv2.WINDOW_NORMAL)        # Mask window (single-lesion) for saliency analaysis
        cv2.moveWindow("MASWindow", 405,360)

        
        # Text defaults for images
        font          = cv2.FONT_HERSHEY_SIMPLEX
        CornerOfText  = (10,20)
        fontScale     = 0.75
        fontColor     = (255,255,255)
        lineType      = 1
        
        # Box line settings
        bbox_line_width = 25
    
    # Make copy for marking and Get dimensions (height, width) of wide field image
    marked_wf_orig_image = wf_orig_image.copy()
    wf_orig_img_height, wf_orig_img_width = wf_orig_image.shape[:2]
    #Create Blank RGB and Grayscalemontages
    wf_montage_RGB_image = np.ones((wf_orig_img_height,wf_orig_img_width,3),np.uint8)
    wf_montage_BW_image = np.zeros((wf_orig_img_height,wf_orig_img_width,3),np.uint8)  # Create montage for size, shape combination saliency analysis
    
    # Fill montage image with image average color(set each pixel to the same value)
    avg_RGB = np.uint8(np.mean(wf_orig_image, axis=(0, 1)))
    wf_montage_RGB_image[:] = (avg_RGB[0], avg_RGB[1], avg_RGB[2])  # Create montage for color, size, shape combination saliency analysis
    
    #Initialize Heatmaps for macro image
    wf_conv_heatmap = np.zeros((wf_orig_img_height,wf_orig_img_width,3),np.uint8) #Convolutional ACTIVATION HEATMAP
    
    # Extract SWIFT Blobs as seeds for SPL analysis and display
    (ROI_blobs, n_blobs, im_with_keypoints) = get_multiscale_moles(wf_orig_image, CLAHE_Adj = False)

    # Adjust image if it is not mostly skin,  naive algorithm (0.9 is the threshold)
    skin_percent = np.sum(skin_detector(wf_orig_image).astype(int))/(wf_orig_img_height*wf_orig_img_width*255)
    print("Skin percentage in image: " + str(skin_percent))
    if(skin_percent>0.75)and(skin_percent<=0.8):
        print("Adjusting")
        wf_orig_image = color_balance(wf_orig_image, 1)
        wf_orig_image = adjust_gamma(wf_orig_image)
        wf_orig_image = apply_clahe(wf_orig_image, c_lim=1.0)
    elif (skin_percent<=0.75):
        print("Adjusting")
        #wf_orig_image = color_balance(wf_orig_image, 1)
        #wf_orig_image = adjust_gamma(wf_orig_image,gamma=1.75)
        wf_orig_image = apply_clahe(wf_orig_image, c_lim=0.25)
        
    elif (skin_percent>0.8):
        print("Adjusting")
        #wf_orig_image = color_balance(wf_orig_image, 1)

    
    if display_plots==True:
        cv2.imshow("SLAWindow", wf_orig_image)
        cv2.imshow("MoleWindow", im_with_keypoints)
    
    # Loop over for each pigmented lesion for analysis
    n_splf = 0 #Counter of non-malignant SPLs to follow
    n_splm = 0 #Counter of possibly malignant SPLs
    im_pls = []  # initialize the list of pigmented lesion image
    f_win =1.5
    
    # # Delete any previous files in Blobs temporary folder
    # blob_temp_folder_path = 'output/analysis/Ugly_Duckling_Analysis/Blobs/'
    # if os.path.exists(blob_temp_folder_path):
    #     shutil.rmtree(blob_temp_folder_path)
    # os.makedirs(blob_temp_folder_path)
    # # Delete any previous files in PLs temporary folder
    # pl_temp_folder_path = 'output/analysis/Ugly_Duckling_Analysis/Pigmented_Lesions/'
    # if os.path.exists(pl_temp_folder_path):
    #     shutil.rmtree(pl_temp_folder_path)
    # os.makedirs(pl_temp_folder_path)
    
    orig_coordinates = np.empty((0,4))
    resized_coordinates = np.empty((0,4))
    ROI_PLs = np.empty((0,3))
    n_blob_prop= np.empty((0,1))
    
    for blob_id in range(0, n_blobs):
        # Get centroid coordinates and diameter of each pigmented lesion (PL) and calculate bounding box x0,x1,y0,y1
        (c_x, c_y, c_d) = ROI_blobs[blob_id,:]
        # We make every bounding box 3x the diameter of the lesion to account for high eccentricity
        x0 = np.uint64(max(0, c_x-f_win*c_d))
        y0 = np.uint64(max(0, c_y-f_win*c_d))
        x1 = np.uint64(max(0, c_x+f_win*c_d))
        y1 = np.uint64(max(0, c_y+f_win*c_d))
        orig_coordinates = np.vstack((orig_coordinates, np.array([x0, y0, x1, y1])))
        
        # Crop PL over wide field image
        crop_img = wf_orig_image[y0:y1, x0:x1] 
        
        # # Save Blob images for later analysis
        # crop_blob_img_file_path = blob_temp_folder_path + 'B_' + str(blob_id) + '.png' 
        # cv2.imwrite(crop_blob_img_file_path,crop_img)
        
        #Get image crop size
        (crop_img_width , crop_img_height) = crop_img.shape[:2]
        
        # Create RGB crop with unmodified lesion segmentation
        masked_crop_RGB_img = np.zeros((crop_img_width,crop_img_height,3),np.uint8)

        # Resize image
        eval_img = cv2.resize(crop_img,(im_dim[0], im_dim[1]))
        
        # Extract SWIFT Blobs as seeds for SPL analysis and display
        (eval_img_ROI_blobs, eval_img_n_blobs, eval_img_im_with_keypoints) = get_center_mole(eval_img, CLAHE_Adj = False)
        #if eval_img_n_blobs > 0:
        #    np.max(eval_img_ROI_blobs[:,2])

        #Classify pigmented lesion (analyze shot, classify and display Convolutional heatmap with class )
        img_tensor = img_to_array(eval_img)                    # (height, width, channel
        img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
        img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
        img_scores = model.predict(img_tensor)
        predicted_class = img_scores.argmax(axis=-1)
        
        #Check if skin is detected by naive algorithm (0.001 is the threshold)
        skin_mole_percent = np.sum(skin_detector(eval_img).astype(int))/(im_dim[0]*im_dim[1]*255)
        #print(img_scores)
        #print(skin_mole_percent)
        
        ## *************************************** SKIN/MOLE CHECK SECTION ********************************* 
        #Check if skin/mole is detected by confirmatory naive algorithm (to reduce false positives)
        th_1=1.0
        th_2=0.995
        if predicted_class==0:
            if (skin_mole_percent>=th_1) and (eval_img_n_blobs>=1):
                print('changed from ' + str(predicted_class) + ' to ')
                predicted_class = np.array([3])
        
        elif predicted_class==1:
            if (skin_mole_percent>=th_2) and (eval_img_n_blobs>=1):
                print('changed from ' + str(predicted_class) + ' to ')
                predicted_class = np.array([3])
                
        elif predicted_class==2:
            if (skin_mole_percent>=th_1) and (eval_img_n_blobs>=1):
                print('changed from ' + str(predicted_class) + ' to ')
                predicted_class = np.array([3])

        elif predicted_class>=3:
            if (eval_img_n_blobs<1):
                print('changed from ' + str(predicted_class) + ' to ')
                predicted_class = np.array([2])
                
            elif (eval_img_n_blobs>=1):
                if (skin_mole_percent<=th_1/8):
                    print('changed from ' + str(predicted_class) + ' to ')
                    predicted_class = np.array([0])
                elif (skin_mole_percent<=th_1/2):
                    print('changed from ' + str(predicted_class) + ' to ')
                    predicted_class = np.array([1])
  
        #print(predicted_class)
        ## *************************************** END OF SECTION ******************************
        
        
        # Display the Macro Window of the sliding process
        if predicted_class == 4:
            n_splf +=1
            marked_wf_orig_image = marked_wf_orig_image.copy()
            if display_plots:
                cv2.rectangle(marked_wf_orig_image, (x0, y0), (x1, y1), (0, 255, 255), bbox_line_width)
                cv2.imshow("SLAWindow", marked_wf_orig_image)
        elif predicted_class == 5:
            n_splm +=1
            marked_wf_orig_image = marked_wf_orig_image.copy()
            if display_plots:
                cv2.rectangle(marked_wf_orig_image, (x0, y0), (x1, y1), (0, 0, 255), bbox_line_width)
                cv2.imshow("SLAWindow", marked_wf_orig_image)
        
        if predicted_class >=3:
            # # Save only potential PL images for later analysis
            # crop_pl_img_file_path = pl_temp_folder_path + 'P_' + str(blob_id) + '.png' 
            # cv2.imwrite(crop_pl_img_file_path,crop_img)
            
            #Populate new ROI_PLs variable with ROI_blobs values of selected pigmented lesions
            ROI_PLs = np.vstack((ROI_PLs, ROI_blobs[blob_id]))
            
            #Track blob size
            if eval_img_n_blobs>0:
                n_blob_prop = np.vstack((n_blob_prop, (eval_img_ROI_blobs[0,2]/(im_dim[0]*im_dim[1]))*100))
            else:
                n_blob_prop = np.vstack((n_blob_prop, 0))
            
            #Append images
            im_pls.append(crop_img)
            crop_gray = cv2.cvtColor(crop_img,cv2.COLOR_BGR2HSV)[:,:,1] #Select saturation channel which is great for skn detection
        
            # Otsu's thresholding with optiona Gaussian filtering
            # crop_blur = cv2.GaussianBlur(crop_gray,(5,5),0)
            thres, mask = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                        
            # Fill RGB crop with unmodified lesion segmentation
            masked_crop_RGB_img = cv2.bitwise_and(crop_img, crop_img, mask=mask)
            # Add segmented RGB crop of original size to montage for compound saliency
            wf_montage_RGB_image[y0:y1,x0:x1,:] = cv2.bitwise_and(wf_montage_RGB_image[y0:y1,x0:x1,:], wf_montage_RGB_image[y0:y1,x0:x1,:], mask = cv2.bitwise_not(mask)) + masked_crop_RGB_img
            
            # Create BW crop with re-scaled binary (0 or 255) lesion segmentation
            masked_crop_BW_img = np.copy(masked_crop_RGB_img)      # Clone RGB crop imag
            masked_crop_BW_img[masked_crop_BW_img > 0] = 255
            # Add segmented and Resized BW crop to montage for shape-only saliency
            c_rd = round(np.mean(ROI_blobs[:,2]))
            rx0 = np.uint64(max(0, c_x-f_win*c_rd))
            ry0 = np.uint64(max(0, c_y-f_win*c_rd))
            rx1 = np.uint64(max(0, c_x+f_win*c_rd))
            ry1 = np.uint64(max(0, c_y+f_win*c_rd))
            resized_coordinates = np.vstack((resized_coordinates, np.array([rx0, ry0, rx1, ry1])))
            im_rdim = wf_montage_BW_image[ry0:ry1,rx0:rx1,:].shape
            
            masked_crop_BW_resize_img = cv2.resize(masked_crop_BW_img,(im_rdim[1],im_rdim[0])) # Re-scale to accelerate  
            wf_montage_BW_image[ry0:ry1,rx0:rx1,:] = wf_montage_BW_image[ry0:ry1,rx0:rx1,:] + masked_crop_BW_resize_img


    return (n_splm, n_splf, ROI_PLs, n_blobs, n_blob_prop, marked_wf_orig_image, im_with_keypoints, im_pls, wf_montage_RGB_image, wf_montage_BW_image)

# Define wide-field feature embedding PL outlier analysis (Ugly Duckling)
def wide_field_feature_embedding_saliency_analysis(wf_orig_image, wf_montage_BW_image, ROI_PLs, embedding_results):
    ## SALIENCY CODE
    # Inputs: 
    #    "wf_orig_image" is the original RGB image
    
    fes_img = wf_montage_BW_image.copy()*0
    cmap=plt.cm.jet
    #Iterate over pigmented lesions and paint over color given the cosine distance from the cnn features
    for index, row in embedding_results.iterrows():
        (x,y,r) = np.uint(ROI_PLs[index])
        color = (np.asarray(cmap(embedding_results.rescaled_scores.values[index]))*255)
        fes_img = cv2.circle(fes_img,(x,y), r, color, -1)
        #fes_img = cv2.cvtColor(fes_img, cv2.COLOR_BGR2RGB)
        (im_width, im_height) = fes_img.shape[:2]
        r = 100.0 / im_height
        dim = (100, int(im_width * r))
        res_fes_img = cv2.resize(fes_img, dim, interpolation = cv2.INTER_AREA)
        res_fes_img = cv2.GaussianBlur(res_fes_img,(5,5),0)
        feature_embedding_saliency_img = cv2.resize(res_fes_img, (im_height, im_width), interpolation = cv2.INTER_CUBIC)
        # Merge Wide field image with heatmap
        wf_feature_embedding_overlay_montage_RGB_image = cv2.addWeighted(wf_orig_image, 0.75, feature_embedding_saliency_img, 0.75, 0)
    return wf_feature_embedding_overlay_montage_RGB_image, feature_embedding_saliency_img

def wide_field_ugly_duckling_analysis(img, cnn_classifer, cnn_feature_extractor):
    wf_orig_image = img

    print(img.shape)

    # Perform multiscale spl id and classification using OPENCV's blob detection and CNN classifier
    (n_splm, n_splf, ROI_PLs, n_blobs, n_blob_prop, marked_wf_orig_image, im_with_keypoints, im_pls, wf_montage_RGB_image, wf_montage_BW_image) = multiscale_wide_field_spl_extraction(wf_orig_image, cnn_classifer, im_dim=[img_width, img_height], layer_name ='block5_conv3', display_plots=False)
    # pickle.dump((n_splm, n_splf, ROI_PLs, n_blobs, n_blob_prop, wf_orig_image, marked_wf_orig_image, im_with_keypoints, im_pls, wf_montage_RGB_image, wf_montage_BW_image), open('output/analysis/Ugly_Duckling_Analysis/multiscale_wide_field_spl_extraction.p', 'wb'))

    # Extract features for all images
    features = []
    for img in im_pls:
        x = cv2.resize(img,(224, 224)) # resize for feature extractor
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feat = cnn_feature_extractor.predict(x)[0]
        features.append(feat)

    #Add relative size of image as part of vector
    min_max_scaler = MinMaxScaler()
    pl_size_norm = min_max_scaler.fit_transform(np.expand_dims(ROI_PLs[:,2], axis=1))* np.max(features)
    features=np.append(features, pl_size_norm,axis=1)

    # Run a PCA analysis on features as example for possible handling with 300 dimensions
    features = np.array(features)
    pca = PCA()
    pca.fit(features)
    pca_features = pca.transform(features)

    # Find mean of the dataset by finding the point with corresponding coordinate means of each feature for entire dataset
    origin = np.array([np.mean(features, axis=0)])

    # Measure distance between origin and all the sample points
    pairwise_dist = distance.cdist(features, origin, metric='cosine')

    # Adjust distance using relative size of lesion as 1/4 the components of the ABCD criteria
    pairwise_dist = min_max_scaler.fit_transform(pairwise_dist)
    pl_size_norm = min_max_scaler.fit_transform(pl_size_norm)-0.5
    n_blob_prop = (min_max_scaler.fit_transform(n_blob_prop)-1.0)
    pairwise_dist = min_max_scaler.fit_transform(pairwise_dist/3 + pl_size_norm/3 + n_blob_prop/3)

    # Extract features in right data type
    odd_scores = np.float64(pairwise_dist)
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 255)).fit(odd_scores)
    rescaled_distances = np.uint8(scaler.transform(odd_scores))
    uint8_odd_scores=[odd_scores[0] for odd_scores in rescaled_distances]

    # Transform into Pandas dataframe to make working with the data easier
    embedding_results = pd.DataFrame({'distance':list(pairwise_dist), 'rescaled_scores':list(uint8_odd_scores)})

    # Sort values by descending distance
    sorted_embedding_results = embedding_results.sort_values('distance', ascending=False)
    print(sorted_embedding_results)

    #Process CNN based ugly duckling image
    wf_feature_embedding_overlay_montage_RGB_image, feature_embedding_saliency_img = wide_field_feature_embedding_saliency_analysis(wf_orig_image, wf_montage_BW_image, ROI_PLs, embedding_results)

    #CLAHE
    wf_feature_embedding_overlay_montage_RGB_image = apply_clahe(wf_feature_embedding_overlay_montage_RGB_image)

    #Save with colormap
    ranked_ugly_ducking_path = 'CNN_ugly_duckling_img.png'

    #Save with colorbar
    cv2.imwrite(ranked_ugly_ducking_path, wf_feature_embedding_overlay_montage_RGB_image)


def main():
    # Load the wide field image
    img_path ='../data/wide_field_images/camera0_051.JPG'
    img = cv2.imread(img_path)

    # load vgg model for classifier
    cnn_classifier_path = '/home/weilunhuang/temp/SPL_UD_DL/Models/vgg16/finetuning_vgg16_cnn_100_epochs.h5'
    cnn_classifier = load_model(cnn_classifier_path)

    # vgg model for feature extractor
    vgg_model = applications.vgg16.VGG16(include_top=True, weights='imagenet')
    cnn_feature_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("fc2").output)

    wide_field_ugly_duckling_analysis(img, cnn_classifier, cnn_feature_extractor)

if __name__ == "__main__":
    main()