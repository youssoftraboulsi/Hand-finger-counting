# Hand-finger-counting
We present a real-time finger counting project using Python (OpenCV and NumPy).

The developed system quickly recognizes the hand fingers being help up and counts them from a live video input. 

The main steps of the project are:

Step 1: select the region of interest

Step 2: extract the hand and separate it from the background using watershed algorithm

Step 3: create a new image in which the fingers being held up are segmented. This was performed using a sliding window approach as follows: for each pixel of the ROI, the sum of pixels of the right half of the window are subtracted from those of its left half

Step 4: denoise the image using opening morphological operation

Step 5: count fingers using contours detection algorithm

