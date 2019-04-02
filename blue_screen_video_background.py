import numpy as np
import time
import cv2

# Start video capture from camera at 640x360 resolution
cap = cv2.VideoCapture(0)
ret = cap.set(3,640)
ret = cap.set(4,360)

# Open video file for background (must be 640x360 resolution)
vid = cv2.VideoCapture('C:/Users/Gustavo/Desktop/100 M Female Athletissima 2012-1.m4v')
frame_counter = 0
fps = vid.get(cv2.CAP_PROP_FPS)
ms=1

# Define the video codec and open a file for saving the resulting video
# frames per second (fps) are the same as the background video
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('C:/Users/Gustavo/Desktop/output.avi',fourcc, fps, (640,360))

while(True):
    
    # Read background frame from video file
    ret1, background = vid.read()
    frame_counter += 1
    
    #If the last frame is reached, reset the capture and the frame_counter
    if frame_counter == vid.get(cv2.CAP_PROP_FRAME_COUNT)-1:
        frame_counter = 0
        vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
    # Capture frame-by-frame
    ret2, frame = cap.read()

    if (ret1 and ret2)==True:    # if both images were captured/loaded correctly
    
        # Store time at the start to later match frame rate
        start = time.time()
        
        # Convert captured frame from BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
        # define range of blue color in HSV
        lower_blue = np.array([90,50,50])
        upper_blue = np.array([150,255,255])
    
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
        # Combine foreground and background using the mask
        output_image = frame.copy()
        output_image[np.where(mask != 0)] = background[np.where(mask != 0)]
        
        # Display the resulting image
        cv2.putText(output_image, str(ms) ,(10,350), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)
        cv2.imshow('output video', output_image)
        #cv2.imshow('input video', frame)
        
        # Save frame
        out.write(output_image)
    
        # Wait some milliseconds to match frame rate, and exit if 'q' key is pressed
        # If more time than required has elapsed, wait only 1 ms
        elapsed = (time.time() - start)*1000
        ms = max(int(1000./float(fps))-elapsed,1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    else:
        break
        
# When everything done, release the captures
cap.release()
vid.release()
out.release()
cv2.destroyAllWindows()