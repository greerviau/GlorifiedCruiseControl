import numpy as np
import cv2
import os
import pickle
import time
import math
import glob
import warnings 
warnings.simplefilter('ignore', np.RankWarning)


import SCNN_lanenet.test_lanenet_raw as ln

lanenet = ln.Lanenet('SCNN_lanenet/model_culane-71-3/culane_lanenet_vgg_2018-12-01-14-38-37.ckpt-10000', True)

def calibrate(size=(1280,720)):
    # Prepare object points 0,0,0 ... 8,5,0
    obj_pts = np.zeros((6*9,3), np.float32)
    obj_pts[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Stores all object points & img points from all images
    objpoints = []
    imgpoints = []

    # Get directory for all calibration images
    images = glob.glob('camera_cal/*.jpg')

    for indx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, size, interpolation = cv2.INTER_AREA)

        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        if ret == True:
            objpoints.append(obj_pts)
            imgpoints.append(corners)
    # Test undistortion on img
    img_size = (img.shape[1], img.shape[0])

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None,None)

    dst = cv2.undistort(img, mtx, dist, None, mtx)
    # Save camera calibration for later use
    dist_pickle = {}
    dist_pickle['mtx'] = mtx
    dist_pickle['dist'] = dist
    pickle.dump( dist_pickle, open('camera_calib.p', 'wb') )

def undistort(img, cal_dir='camera_calib.p'):
    with open(cal_dir, mode='rb') as f:
        file = pickle.load(f)
    mtx = file['mtx']
    dist = file['dist']
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    
    return dst

def color_grad_thresh(img, rgb_thresh=(150,255), l_thresh=(80,255), sobel_thresh=(25,256), dir_thresh=(0.7, 1.4)):

    area = img.shape[0] * img.shape[1]

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    #cv2.imshow('gray', gray)

    #REMOVE SHADOWS (AS BEST AS POSSIBLE)
    #m = np.max(gray)
    #mask = np.array(np.power(2, (255-gray)/34.5), dtype=np.uint8)
    #cv2.imshow('mask', mask)
    #gray = gray + mask
    #gray = cv2.GaussianBlur(gray, (5,5), 0)

    #cv2.imshow('no shadows', gray)

    #RGB COLOR SPACE
    r_channel = img[:,:,0].copy()
    g_channel = img[:,:,1].copy()
    b_channel = img[:,:,2].copy()

    '''
    cv2.imshow('r channel', r_channel)
    cv2.imshow('g channel', g_channel)

    r_channel[(r_channel >= 20) & (r_channel <= 60)] *= 3
    g_channel[(g_channel >= 20) & (g_channel <= 60)] *= 3

    cv2.imshow('r channel boost', r_channel)
    cv2.imshow('g channel boost', g_channel)
    '''

    r_binary = np.zeros_like(r_channel)
    r_binary[(r_channel >= np.max(r_channel)-30)] = 1
    #cv2.imshow('r bin', r_binary*255)

    g_binary = np.zeros_like(g_channel)
    g_binary[(g_channel >= np.max(g_channel)-30)] = 1
    #cv2.imshow('g bin', g_binary*255)

    '''
    #LAB COLOR SPACE
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB) 
    l_channel = lab[:,:,0]
    a_channel = lab[:,:,1]
    b_channel = lab[:,:,2]
    #cv2.imshow('l', l_channel)
    #cv2.imshow('a', a_channel)
    #cv2.imshow('b', b_channel)
    '''

    # HSV COLOR SPACE
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h_channel = hsv[:,:,0]
    s_channel = hsv[:,:,1]
    v_channel = hsv[:,:,2]
    #cv2.imshow('hsv', hsv)
    #cv2.imshow('h', h_channel)
    #cv2.imshow('s', s_channel)
    #cv2.imshow('v', v_channel)

    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= np.max(v_channel)-30)] = 1
    #cv2.imshow('v bin', v_binary*255)

    # YCRCB COLOR SPACE
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y_channel = ycrcb[:,:,0]
    cr_channel = ycrcb[:,:,1]
    cb_channel = ycrcb[:,:,2]
    #cv2.imshow('cr', cr_channel)
    #cv2.imshow('cb', cb_channel)

    cr_binary = np.zeros_like(cr_channel)
    cr_binary[(cr_channel >= np.mean(cr_channel)+10)] = 1
    #cv2.imshow('cr bin', cr_binary*255)

    cb_binary = np.zeros_like(cb_channel)
    cb_binary[(cb_channel <= np.mean(cb_channel)-10)] = 1
    #cv2.imshow('cb bin', cb_binary*255)
    '''
    # SOBEL EDGE DETECTION

    sobel_y = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    sobel_x = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))

    sobel_scaled_y = np.uint8(255*sobel_y/np.max(sobel_y))
    sobel_scaled_x = np.uint8(255*sobel_x/np.max(sobel_x))

    sobel_binary = np.zeros_like(sobel_scaled_x)
    sobel_binary[(sobel_scaled_x > sobel_thresh[0]) & (sobel_scaled_x <= sobel_thresh[1])] = 1
    sobel_binary[(sobel_scaled_y > sobel_thresh[0]) & (sobel_scaled_y <= sobel_thresh[1])] = 1
    #cv2.imshow('sobel', sobel_binary*255)
    '''
    if np.sum(g_binary) > area/12: 
        g_binary = np.zeros_like(g_binary)
    if np.sum(r_binary) > area/12: 
        r_binary = np.zeros_like(r_binary)
    if np.sum(v_binary) > area/12: 
        v_binary = np.zeros_like(v_binary)
    if np.sum(cr_binary) > area/12: 
        cr_binary = np.zeros_like(cr_binary)
    if np.sum(cb_binary) > area/12: 
        cb_binary = np.zeros_like(cb_binary)

    combined_binary = np.zeros_like(gray)
    combined_binary[(g_binary == 1) | (r_binary == 1) | (v_binary == 1) | (cr_binary == 1) | (cb_binary == 1)] = 1
    return combined_binary


def hls_compute_binary(img, s_thresh=(100, 255), sx_thresh=(15, 255)):
    img = cv2.GaussianBlur(img, (5,5), 0)

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    h_channel = hls[:,:,0]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary

def perspective_warp(img, roi, dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])):
    src = np.float32([(roi[0],0), (roi[1], 0), (roi[2], 1), (roi[3], 1)])
    dst_size = (img.shape[1], img.shape[0])
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src * img_size
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped

def inv_perspective_warp(img, roi, src=np.float32([(0,0), (1, 0), (0,1), (1,1)])):
    dst = np.float32([(roi[0],0), (roi[1], 0), (roi[2], 1), (roi[3], 1)])
    dst_size = (img.shape[1], img.shape[0])
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src * img_size
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped

def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist


def sliding_window(img, nwindows=6, margin=100, minpix = 1, draw_windows=True):
    left_a, left_b, left_c = [],[],[]
    right_a, right_b, right_c = [],[],[]
    left_fit_= np.empty(3)
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img))*255

    histogram = get_hist(img)
    # Find peaks of left and right halves
    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    
    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        if draw_windows == True:
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (100,255,255), 3) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (100,255,255), 3) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
        
#        if len(good_right_inds) > minpix:        
#            rightx_current = np.int(np.mean([leftx_current +900, np.mean(nonzerox[good_right_inds])]))
#        elif len(good_left_inds) > minpix:
#            rightx_current = np.int(np.mean([np.mean(nonzerox[good_left_inds]) +900, rightx_current]))
#        if len(good_left_inds) > minpix:
#            leftx_current = np.int(np.mean([rightx_current -900, np.mean(nonzerox[good_left_inds])]))
#        elif len(good_right_inds) > minpix:
#            leftx_current = np.int(np.mean([np.mean(nonzerox[good_right_inds]) -900, leftx_current]))


    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    left_a.append(left_fit[0])
    left_b.append(left_fit[1])
    left_c.append(left_fit[2])
    
    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])
    
    left_fit_[0] = np.mean(left_a[-10:])
    left_fit_[1] = np.mean(left_b[-10:])
    left_fit_[2] = np.mean(left_c[-10:])
    
    right_fit_[0] = np.mean(right_a[-10:])
    right_fit_[1] = np.mean(right_b[-10:])
    right_fit_[2] = np.mean(right_c[-10:])
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit_[0]*ploty**2 + left_fit_[1]*ploty + left_fit_[2]
    right_fitx = right_fit_[0]*ploty**2 + right_fit_[1]*ploty + right_fit_[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]
    
    return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty

def get_curve(img, leftx, rightx):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    y_eval = np.max(ploty)
    ym_per_pix = 30.5/720 # Meters per pixel in y dimension
    xm_per_pix = 3.7/720 # Meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    car_pos = img.shape[1]/2
    l_fit_x_int = left_fit_cr[0]*img.shape[0]**2 + left_fit_cr[1]*img.shape[0] + left_fit_cr[2]
    r_fit_x_int = right_fit_cr[0]*img.shape[0]**2 + right_fit_cr[1]*img.shape[0] + right_fit_cr[2]
    lane_center_position = (r_fit_x_int - l_fit_x_int) /2+l_fit_x_int
    offset = (car_pos - lane_center_position) * xm_per_pix / 10
    # Now our radius of curvature is in meters
    return (left_curverad, right_curverad, offset)

def draw_lanes(img, left_fit, right_fit, roi):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    color_img = np.zeros_like(img)
    
    left = np.array(np.transpose(np.vstack([left_fit, ploty])), dtype=np.int)
    right = np.array(np.transpose(np.vstack([right_fit, ploty])), dtype=np.int)
    center = np.mean([left,right],axis=0, dtype=np.int)
    points = np.vstack((left, np.flip(right, 0)))
    
    #cv2.fillPoly(color_img, [points], (0,255,0))
    cv2.polylines(color_img, [left], False, (255,255,255), 10)
    cv2.polylines(color_img, [right], False, (255,255,255), 10)
    cv2.polylines(color_img, [center], False, (0,255,0), 40)
    inv_perspective = inv_perspective_warp(color_img, roi=roi)

    return inv_perspective

def vid_pipeline(img, cache, roi, write=True, show=True):
    image = np.copy(img)
    font = cv2.FONT_HERSHEY_PLAIN
    fontColor = (0, 0, 0)
    fontSize=1.7
    thickness = 2
    size = (image.shape[1],image.shape[0])

    #image = undistort(image)

    #cv2.imwrite('current_frame.png', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    find_lanes = lanenet.run_lanenet(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    scale_factor = 255 / find_lanes.max()
    find_lanes = find_lanes * scale_factor
    find_lanes = np.array(find_lanes, dtype=np.uint8)
    find_lanes = cv2.resize(find_lanes, size)
    #cv2.imshow('find', find_lanes)
    #cv2.imshow('lanes', np.maximum(image, find_lanes))
    #print(type(find_lanes))
    #print(find_lanes.shape)
    #print(size)
    lanes_slice = find_lanes[math.floor(size[1]*roi[0][1]):math.floor(size[1]*roi[3][1]), :] # Create the ROI slice of the frame
    #cv2.imshow('slice', lanes_slice)
    lanenet_perspect = perspective_warp(lanes_slice, roi=[roi[0][0], roi[1][0], roi[2][0], roi[3][0]]) # Warp the ROI to birdseye view
    #cv2.imshow('persp', perspect)    
    lanenet_red = lanenet_perspect[:,:,2].copy()
    lanenet_green = lanenet_perspect[:,:,1]
    #lanes_thresh = np.maximum(lanenet_perspect[:,:,2], lanenet_perspect[:,:,1])
    #cv2.imshow('gray', lanes_thresh)
    lanenet_red_thresh = np.zeros_like(lanenet_red)
    lanenet_red_thresh[lanenet_red > np.max(lanenet_red)/1.5] = 1

    lanenet_green_thresh = np.zeros_like(lanenet_green)
    lanenet_green_thresh[lanenet_green > np.max(lanenet_green)/1.5] = 1

    lanenet_thresh = (lanenet_green_thresh + lanenet_red_thresh).copy()
    #cv2.imshow('thresh', thresh*255)
    #cv2.waitKey(0)
    
    img_slice = image[math.floor(size[1]*roi[0][1]):math.floor(size[1]*roi[3][1]), :] # Create the ROI slice of the frame
    
    #pipe = hls_compute_binary(img_slice)  # Run initial video pipeline
    
    color_perspect = perspective_warp(img_slice, roi=[roi[0][0], roi[1][0], roi[2][0], roi[3][0]]) # Warp the ROI to birdseye view

    color_thresh = color_grad_thresh(color_perspect)

    thresh = np.zeros_like(color_thresh)
    thresh[(color_thresh == 1) | (lanenet_thresh == 1)] = 1
    
    #thresh = lanenet_thresh
    # Run sliding windows to get the curve of each lane marker
    try:    
        sliding, curves, _, _ = sliding_window(thresh, margin=100, nwindows=4, draw_windows=True)

        left_angle = np.rad2deg(np.arctan2(len(curves[0]), curves[0][-1] - curves[0][0]))
        right_angle = np.rad2deg(np.arctan2(len(curves[1]), curves[1][-1] - curves[1][0]))
        center_angle = np.mean([left_angle, right_angle])

        #print(center_angle)

        curverad = get_curve(img_slice, curves[0], curves[1])   # Calculate the curve radius of each lane line

        left_curve = curverad[0]
        right_curve = curverad[1]
        vehicle_offset = curverad[2]

        if left_curve  > 5000 or right_curve > 5000:
            raise Exception

        lane_curve = np.mean([left_curve, right_curve])    # Calculate the average radius

        #COMMENT OUT IF USING MEDIAN
        #///////////////////////////////

        mean_left_curve = left_curve
        mean_right_curve = right_curve
        mean_vehicle_offset = vehicle_offset
        mean_lane_curve = lane_curve
        mean_center_angle = center_angle

        median_left_curve = left_curve
        median_right_curve = right_curve
        median_vehicle_offset = vehicle_offset
        median_lane_curve = lane_curve
        median_center_angle = center_angle

        #last_lane_curve = lane_curve
        last_center_angle = center_angle
        
        if not cache.empty():
            _, _, _, _, _, _, last_lane_curve, last_center_angle, _, _, _, _, _, _, _ = cache.get_last()
            #MEAN CALCULATIONS
            mean_left_curve = cache.mean(3)
            mean_left_curve = np.mean([mean_left_curve, left_curve])

            mean_right_curve = cache.mean(4)
            mean_right_curve = np.mean([mean_right_curve, right_curve])

            mean_vehicle_offset = cache.mean(5)
            mean_vehicle_offset = np.mean([mean_vehicle_offset, vehicle_offset])

            mean_lane_curve = cache.mean(6)
            mean_lane_curve = np.mean([mean_lane_curve, lane_curve])

            mean_center_angle = cache.mean(7)
            mean_center_angle = np.mean([mean_center_angle, center_angle])
            
            #MEDIAN CALCULATIONS
            median_left_curve = cache.median(3)
            median_left_curve = np.median([median_left_curve, left_curve])

            median_right_curve = cache.median(4)
            median_right_curve = np.median([median_right_curve, right_curve])

            median_vehicle_offset = cache.median(5)
            median_vehicle_offset = np.median([median_vehicle_offset, vehicle_offset])

            median_lane_curve = cache.median(6)
            median_lane_curve = np.median([median_lane_curve, lane_curve])

            median_center_angle = cache.median(7)
            median_center_angle = np.median([median_center_angle, center_angle])

            curves = np.mean(np.stack([np.array(cache.mean(2)), np.array(curves)]),axis=0)
            curves = list(curves)
            
        #///////////////////////////////

        turn = 'straight'

        #print(mean_center_angle)

        if mean_center_angle < 80.0:
            turn = 'left'
        elif mean_center_angle > 110.0:
            turn = 'right'

        # Get the lane polygon
        lanes = draw_lanes(img_slice, curves[0], curves[1], roi=[roi[0][0], roi[1][0], roi[2][0], roi[3][0]])
        #print(abs(last_center_angle - mean_center_angle))
        '''
        if abs(last_center_angle - mean_center_angle) > 15:
            mean_curverad = cache.mean(3)
            mean_lane_curve = cache.mean(4)
            mean_center_angle = cache.mean(5)

        if abs(last_center_angle - median_center_angle) > 15:
            median_curverad = cache.median(3)
            median_lane_curve = cache.median(4)
            median_center_angle = cache.median(5)
        '''
        cache.add([roi.copy(), np.copy(sliding), np.copy(curves), mean_left_curve, mean_right_curve, mean_vehicle_offset, mean_lane_curve, mean_center_angle, median_left_curve, median_right_curve, median_vehicle_offset, median_lane_curve, median_center_angle, turn, np.copy(lanes)])
        '''
        median_idx = cache.median_index(5)
        lanes = cache.get_at_index(median_idx, cache.get_element_size()-1)
        roi = cache.get_at_index(median_idx, 0)'''
    except Exception as ex:
        roi, sliding, curves, mean_left_curve, mean_right_curve, mean_vehicle_offset, mean_lane_curve, mean_center_angle, median_left_curve, median_right_curve, median_vehicle_offset, median_lane_curve, median_center_angle, turn, lanes = cache.get_last()
        
    # Add the lane polygon to the output image
    img_slice = np.copy(image[math.floor(size[1]*roi[0][1]):math.floor(size[1]*roi[3][1]), :])
    #print(lanes.shape, img_slice.shape)
    image[math.floor(size[1]*roi[0][1]):math.floor(size[1]*roi[3][1]), :] = cv2.addWeighted(img_slice, 1, lanes, 0.6, 0)

    image = cv2.resize(image, (1280, 720))
    
    if write:
        cv2.putText(image, 'Mean Left Lane Radius: {:.0f} m'.format(mean_left_curve), (10, 30), font, fontSize, fontColor, thickness)
        cv2.putText(image, 'Median Left Lane Radius: {:.0f} m'.format(median_left_curve), (10, 55), font, fontSize, fontColor, thickness)

        cv2.putText(image, 'Mean Right Lane Radius: {:.0f} m'.format(mean_right_curve), (10, 80), font, fontSize, fontColor, thickness)
        cv2.putText(image, 'Median Right Lane Radius: {:.0f} m'.format(median_right_curve), (10, 105), font, fontSize, fontColor, thickness)

        cv2.putText(image, 'Mean Center Lane Radius: {:.0f} m'.format(mean_lane_curve), (10, 130), font, fontSize, fontColor, thickness)
        cv2.putText(image, 'Median Center Lane Radius: {:.0f} m'.format(median_lane_curve), (10, 155), font, fontSize, fontColor, thickness)

        cv2.putText(image, 'Mean Vehicle Offset: {:.4f} m'.format(mean_vehicle_offset), (10, 180), font, fontSize, fontColor, thickness)
        cv2.putText(image, 'Median Vehicle Offset: {:.4f} m'.format(median_vehicle_offset), (10, 205), font, fontSize, fontColor, thickness)

        cv2.putText(image, 'Mean Center Angle: {:.2f}'.format(mean_center_angle), (10, 230), font, fontSize, fontColor, thickness)  
        cv2.putText(image, 'Median Center Angle: {:.2f}'.format(median_center_angle), (10, 255), font, fontSize, fontColor, thickness)

        cv2.putText(image, 'Turn: {}'.format(turn), (10, 280), font, fontSize, fontColor, thickness)

    image = cv2.resize(image, size)
    # Assemble the visual frame if requested
    visual_frame = None
    if show:
        visual_frame = np.concatenate((cv2.addWeighted(cv2.cvtColor(img_slice, cv2.COLOR_RGB2BGR), 1, lanes_slice, 1, 0),sliding,cv2.cvtColor(lanes, cv2.COLOR_RGB2BGR)), axis=0)

        line_y = thresh.shape[0]
        cv2.line(visual_frame, (0,line_y), (visual_frame.shape[1],line_y), (255,255,255), 2)

        line_y += sliding.shape[0]
        cv2.line(visual_frame, (0,line_y), (visual_frame.shape[1],line_y), (255,255,255), 2)

        line_y+= lanes.shape[0]
        cv2.line(visual_frame, (0,line_y), (visual_frame.shape[1],line_y), (255,255,255), 2)

        visual_frame = cv2.resize(visual_frame, (image.shape[1], image.shape[0]),  interpolation = cv2.INTER_AREA)  

    return np.copy(image), np.copy(visual_frame), ((mean_lane_curve, mean_left_curve, mean_right_curve, mean_vehicle_offset, mean_center_angle), (median_lane_curve, median_left_curve, median_right_curve, median_vehicle_offset, median_center_angle), turn)

