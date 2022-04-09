import cv2

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

# Define wide-field saliency analysis
def wide_field_naive_saliency_analysis(wf_montage_RGB_image, wf_montage_BW_image, marked_wf_orig_image, width=1000):
    ## SALIENCY CODE
    # Inputs: 
    #    "wf_montage_RGB_image" is a background simplified RGB image of moles to assess color and size differences
    #    "wf_montage_BW_image" is a background simplified Binarized image of rescaled moles to asses shape differences
    #    "marked_wf_orig_image" is the wide-field image with previous marks to overlay the results on.
    # Outputs:
    #    "wf_overlay_montage_RGB_image" is an overlayed image with the saliency output
    #    "saliency_img" is the saliency output   
    # Modified from Saliency Code on https://github.com/mayoyamasaki/saliency-map using Laurent Itti, Christof Koch (2000) method
    print('Processing Naive Saliency (Ugly Ducking), this may take a while...', end='')
    spinner = Spinner()
    spinner.start()
    
    # Analize Ugly Duckling (saliency) considering all Class 3 or above pigmented lesions with resizing for speed
    in_sm_c = imutils.resize(wf_montage_RGB_image, width=width)
    #in_sm_C = wf_montage_RGB_image
    sm_c = SaliencyMap(in_sm_c)
    compound_saliency_img = OpencvIo().saliency_array2img([sm_c.map])
    
    # Analize Ugly Duckling (saliency) considering all Class 3 or above reshaped pigmented lesions with resizing for speed
    in_sm_s = imutils.resize(wf_montage_BW_image, width=width)
    #in_sm_s = wf_montage_BW_image
    sm_s = SaliencyMap(in_sm_s)
    shape_saliency_img = OpencvIo().saliency_array2img([sm_s.map])
    
    # Get main image dimensions for overlay
    wf_orig_img_height, wf_orig_img_width = marked_wf_orig_image.shape[:2]
    
    # Merge saliency maps
    saliency_img = cv2.applyColorMap(cv2.addWeighted(compound_saliency_img, 0.75, shape_saliency_img, 0.25, 0), cv2.COLORMAP_JET)
    wf_overlay_montage_RGB_image = cv2.addWeighted(marked_wf_orig_image, 0.5, cv2.resize(saliency_img,(wf_orig_img_width, wf_orig_img_height)), 0.5, 0)
    spinner.stop()
    
    return wf_overlay_montage_RGB_image, saliency_img

### Define multiscale spl id and classification function using OPENCV's  blob detection and CNN classifier
def multiscale_wide_field_spl_analysis(wf_orig_image, model, im_dim=[img_width,img_height], layer_name = 'dense', display_plots=False):
    
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
    
    # TQDM Progressbar
    pbar = tqdm(total=n_blobs)

    # Loop over for each pigmented lesion for analysis
    n_splf = 0 #Counter of non-malignant SPLs to follow
    n_splm = 0 #Counter of possibly malignant SPLs
    im_pls = []  # initialize the list of pigmented lesion image
    f_win =1.5
    
    # Delete any previous files in Blobs temporary folder
    blob_temp_folder_path = 'output/analysis/Ugly_Duckling_Analysis/Blobs/'
    shutil.rmtree(blob_temp_folder_path)
    os.makedirs(blob_temp_folder_path)
    # Delete any previous files in PLs temporary folder
    pl_temp_folder_path = 'output/analysis/Ugly_Duckling_Analysis/Pigmented_Lesions/'
    shutil.rmtree(pl_temp_folder_path)
    os.makedirs(pl_temp_folder_path)
    
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
        
        # Save Blob images for later analysis
        crop_blob_img_file_path = blob_temp_folder_path + 'B_' + str(blob_id) + '.png' 
        cv2.imwrite(crop_blob_img_file_path,crop_img)
        
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
            cv2.rectangle(marked_wf_orig_image, (x0, y0), (x1, y1), (0, 255, 255), bbox_line_width)
            cv2.imshow("SLAWindow", marked_wf_orig_image)
        elif predicted_class == 5:
            n_splm +=1
            marked_wf_orig_image = marked_wf_orig_image.copy()
            cv2.rectangle(marked_wf_orig_image, (x0, y0), (x1, y1), (0, 0, 255), bbox_line_width)
            cv2.imshow("SLAWindow", marked_wf_orig_image)
        
        if predicted_class >=3:
            # Save only potential PL images for later analysis
            crop_pl_img_file_path = pl_temp_folder_path + 'P_' + str(blob_id) + '.png' 
            cv2.imwrite(crop_pl_img_file_path,crop_img)
            
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

        # Process and Display CNN output for each window
        layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]
        eval_conv_heatmap = cv2.cvtColor(visualize_cam(model=model, layer_idx=layer_idx, filter_indices = [None], seed_input = eval_img), cv2.COLOR_BGR2RGB)
        eval_overlay_conv_heatmap = cv2.putText(overlay(eval_img, eval_conv_heatmap),('Class: '+str(predicted_class)), CornerOfText, font, fontScale, fontColor, lineType)
        
        # Construct rescaled heatmap with dimensions (height, width) of original crop  image
        crop_conv_heatmap = cv2.resize(eval_overlay_conv_heatmap,(crop_img_height, crop_img_width))
            
        # Stitch CNN output for macro image display
        wf_conv_heatmap[y0:y1,x0:x1,:] = cv2.addWeighted(wf_conv_heatmap[y0:y1,x0:x1,:], 0.5, crop_conv_heatmap, 0.5,  0)
        wf_overlay_conv_heatmap = cv2.addWeighted(wf_orig_image, 1.0, wf_conv_heatmap, 0.5, 0)
        
        # Display SPL and wide-field CNN outputs
        if display_plots==True:
            cv2.imshow("CAMWindow", eval_overlay_conv_heatmap) # Display class filter with single lesions
            cv2.imshow("MASWindow", masked_crop_RGB_img) # Display mask for shape saliency analysis (masked_crop_RGB_img or eval_img_im_with_keypoints)
            cv2.imshow("SPLWindow", wf_overlay_conv_heatmap)   # Display class filter with all lesions
            cv2.waitKey(1)
            time.sleep(0.025)
        
        pbar.update(1)
    
    # NAIVE SALIENCY FUNCTION
    wf_overlay_montage_RGB_image, saliency_img = wide_field_naive_saliency_analysis(wf_montage_RGB_image, wf_montage_BW_image, marked_wf_orig_image, width=1000)

    # Display  
    if display_plots==True:
        
        #Create saliency windows
        cv2.namedWindow("SALWindow", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
        cv2.moveWindow("SALWindow", 405,695)
        
        # Display Ugly Duckling Analysis (Saliency)
        cv2.imshow('SALWindow',wf_overlay_montage_RGB_image)
        cv2.waitKey(1)
    
    print('Analysis Completed!')
    
    # Close process bar
    pbar.close()
    
    return n_splm, n_splf, ROI_PLs, n_blobs, n_blob_prop, marked_wf_orig_image, wf_overlay_conv_heatmap, im_with_keypoints, im_pls, wf_montage_RGB_image, wf_montage_BW_image, saliency_img, wf_overlay_montage_RGB_image