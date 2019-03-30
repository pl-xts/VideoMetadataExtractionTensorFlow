#!matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

from darkflow.net.build import TFNet
import cv2
import pprint as pp
from darkflow.net.build import TFNet

options = {"model": "cfg/yolo.cfg", 
           "load": "bin/yolo.weights", 
           "threshold": 0.1, 
           "gpu": 1.0,
		   "track": True,
		   "trackObj": ['Bicyclist','Pedestrian','Skateboarder','Cart','Car','Bus'],
		   "tracker": "sort",
		   "display": True}

tfnet = TFNet(options)

def boxing(original_img, predictions):
    newImage = np.copy(original_img)

    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        label = result['label'] + " " + str(round(confidence, 3))

        if confidence > 0.3:
            newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 3)
            newImage = cv2.putText(newImage, label, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, (0, 230, 0), 1, cv2.LINE_AA)
            
    return newImage

cap = cv2.VideoCapture('./sample_video/smaller.mp4')
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('./sample_video/output.avi',fourcc, 20.0, (int(width), int(height)))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret == True:
        frame = np.asarray(frame)
        results = tfnet.return_predict(frame)
        # with open("output.txt", "a") as f:
         # for i in results:
           # f.write('%s\n' % i)
        new_frame = boxing(frame, results)

        # Display the resulting frame
        out.write(new_frame)
        #cv2.imshow('frame',new_frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
    else:
        break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()