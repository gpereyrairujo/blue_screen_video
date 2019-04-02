import numpy as np
import os
import sys
import time
import cv2

# Folder containing background videos
input_path = 'C:/Users/Gustavo/Desktop/videos/'
output_path = 'C:/Users/Gustavo/Desktop/output/'

# Open a window in full screen mode
cv2.namedWindow("Blue Screen", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Blue Screen",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

while(True):

    # Loop through all files in the input folder
    for filename in os.listdir(input_path):
    
        # Start video capture from camera at 640x360 resolution
        cap = cv2.VideoCapture(0)
        ret = cap.set(3,640)
        ret = cap.set(4,360)
        
        # Open video file for background (must be 640x360 resolution)
        vid = cv2.VideoCapture(input_path+filename)
        frame_counter = 0
        fps = vid.get(cv2.CAP_PROP_FPS)
        ms=1
        
        # Define the video codec and open a file for saving the resulting video
        # frames per second (fps) are the same as the background video
        # File name is output+date+time.mp4
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        timestr = time.strftime("-%Y%m%d-%H%M%S")
        out = cv2.VideoWriter(output_path+'output'+timestr+'.mp4',fourcc, fps, (640,360))
        
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
            
            # Flip frame horizontally to mimic a mirror
            frame = cv2.flip(frame,1)
        
            # Process if both images were captured/loaded correctly
            if (ret1 and ret2)==True:    
            
                # Store time at the start to later match frame rate
                start = time.time()
                
                # Convert captured frame from BGR to HSV
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
                # define range of blue color in HSV
                lower_blue = np.array([90,100,50])
                upper_blue = np.array([150,255,255])
            
                # Threshold the HSV image to get only blue colors
                mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
                # Combine foreground and background using the mask
                output_image = frame.copy()
                output_image[np.where(mask != 0)] = background[np.where(mask != 0)]
                
                # Display the resulting image
                cv2.imshow("Blue Screen", output_image)
                
                # Save frame
                out.write(output_image)
            
                # Wait some milliseconds to match frame rate
                # If more time than required has elapsed, wait only 1 ms
                elapsed = (time.time() - start)*1000
                ms = max(int(1000./float(fps))-elapsed,1)
                
                # If space bar is pressed skip to next video
                k = cv2.waitKey(1) & 0xFF 
                if k == ord(' '):
                    break

                # If Q key is pressed release capture/files, close window and exit
                elif k == ord('q'):
                    cap.release()
                    vid.release()
                    out.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)
                    
            # Skip video if any of the images was not captured/loaded correctly
            else:
                break
                
        # When video has been skipped, release the capture/files
        cap.release()
        vid.release()
        out.release()

# Close window
cv2.destroyAllWindows()