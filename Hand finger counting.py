import numpy as np
import cv2

# integral_image method: takes the image as input and returns it's integral image
def integral_image(image):
    new_image = np.zeros((image.shape[0]+1,image.shape[1]+1))
    for i in range(1,new_image.shape[0]):
        for j in range(1,new_image.shape[1]):
            new_image[i,j] = new_image[i-1,j] + new_image[i,j-1] - new_image[i-1,j-1] + image[i-1,j-1]
    
    new_image = np.delete(new_image, 0, 0)
    new_image = np.delete(new_image, 0, 1)
    return new_image

cap = cv2.VideoCapture(0)
while True:
    ret,img = cap.read()
    img = cv2.flip(img,1)
    cv2.putText(img,text="Make sure to include the blue disk in your hand!",org=(10,round(img.shape[0]/1.2)),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=0.7,color=(0,255,0),thickness=1)
    
    # Step 1: select the region of interest
    x1 = img.shape[1]//2
    x2 = round(img.shape[1]*7/8)
    y1 = img.shape[0]//10
    y2 = round(img.shape[0]/1.5)
    
    cv2.rectangle(img, (x1,y1), (x2,y2), color=(0,0,255),thickness=4)
    cv2.imshow('img',img)

    hand = img[y1:y2,x1:x2]
    hand_copy = np.copy(hand)
    
    # Step 2: extract the hand and separate it from the background using watershed algorithm
    marker_image = np.zeros(hand.shape[:2],dtype=np.int32)
    
    x = marker_image.shape[1] // 2
    y = marker_image.shape[0] * 7 // 8
    
    r=20
    cv2.circle(marker_image,(x,y),r,1,-1)
    cv2.circle(hand_copy,(x,y),r,(255,0,0),-1)  
          
    r=1
    cv2.circle(marker_image,(5,5),r,2,-1)

    marker_image_copy = marker_image.copy()
    marker_image_copy = cv2.watershed(hand,marker_image_copy)
    
    segments = np.zeros(hand.shape[:2],dtype=np.uint8)
    
    colors = [0,1,2]
    for color_ind in range(1,3):
        segments[marker_image_copy==color_ind] = colors[color_ind]
    
    
    # Step 3: create a new image in which the fingers being held up are segmented. This was performed using a sliding window approach as follows: for each pixel of the ROI, the sum of pixels of the right half of the window are subtracted from those of its left half
    window_x = 11
    window_y = 21
    
    wx2 = (window_x-1)//2
    wy2 = (window_y-1)//2


    integral_segments = integral_image(segments) 

    M1 = np.zeros((segments.shape[0] + window_x, segments.shape[1] + wy2))
    M2 = np.zeros((segments.shape[0] + window_x, segments.shape[1] + wy2))
    M3 = np.zeros((segments.shape[0] + window_x, segments.shape[1] + wy2))
    M4 = np.zeros((segments.shape[0] + window_x, segments.shape[1] + wy2))
    
    M1[:segments.shape[0],:segments.shape[1]] = integral_segments
    M2[:segments.shape[0],wy2:segments.shape[1]+wy2] = integral_segments
    M3[window_x:segments.shape[0]+window_x,:segments.shape[1]] = integral_segments
    M4[window_x:segments.shape[0]+window_x,wy2:segments.shape[1]+wy2] = integral_segments
    combined_M = M1-M2-M3+M4 
    
    combined_M_right = np.zeros((combined_M.shape[0],combined_M.shape[1]+wy2+1))
    combined_M_right[:,:combined_M.shape[1]] = combined_M
    combined_M_left = np.zeros((combined_M.shape[0],combined_M.shape[1]+wy2+1))
    combined_M_left[:,wy2+1:combined_M.shape[1]+wy2+1] = combined_M
    
    result = combined_M_right-combined_M_left
    result = np.delete(result,np.s_[:window_x],axis=0)
    result = np.delete(result,np.s_[:window_y],axis=1)

    result = np.delete(result,np.s_[result.shape[0] - window_x:],axis=0)
    result = np.delete(result,np.s_[result.shape[1] - window_y:],axis=1)
    cv2.imshow('result',result)
    
    # Step 4: denoise the image using opening morphological operation
    kernel = np.ones((10,10),dtype=np.uint8)
    opening = cv2.morphologyEx(result,cv2.MORPH_OPEN,kernel)
    opening[opening<0]=0
    cv2.imshow('opening',opening)

    # Step 5: count fingers using contours detection algorithm    
    opening = np.uint8(opening)
    contours,hierarchy = cv2.findContours(opening,mode=cv2.RETR_CCOMP,method=cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        ans = 0
    else:
        ans = sum(sum(hierarchy[:,:,3]==-1))
        
    cv2.putText(hand_copy,text=str(ans),org=(hand_copy.shape[0]//20,hand_copy.shape[1]//4),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=1,color=(255,0,0),thickness=4)
    cv2.imshow('hand',hand_copy)

    if cv2.waitKey(20) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()