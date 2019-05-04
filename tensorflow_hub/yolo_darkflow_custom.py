#!matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

from darkflow.net.build import TFNet
import cv2
import pprint as pp
from darkflow.net.build import TFNet

# For measuring the inference time.
import time
from datetime import datetime

from utils import prepare_results as pr
from utils import store_results as sr

# Change main paramateres
path_to_file = "./sample_video/"
video_name = "zoo"
video_type = ".mp4"
model_name = "Yolo V2 Custom"
min_threshold = 10
cap = cv2.VideoCapture(path_to_file + video_name + video_type)
# Properties of video file
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS )
# Only each [procesing_frame_rate] frame will be used for prediction
# 30 fps = every second 1 frame will be used
procesing_frame_rate = fps #int(frame_count / int(frame_count/fps))
# parameter required for process completion track
point = frame_count / 100
# parameters requried in resize()
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
# parameter for enabling frame resizing 
do_resize = False

options = {"model": "./cfg/tiny-yolo-vocCustom.cfg",
           "threshold": 0.1, 
           "gpu": 1.0,
           "load": 700,}
#700, 888, 1332, -1
start = time.time()
tfnet = TFNet(options)
tfnet.load_from_ckpt()

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('./sample_video/output.avi',fourcc, 20.0, (int(width), int(height)))

def boxing(original_img, predictions):
    newImage = np.copy(original_img)

    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        label = result['label'] + " " + str(round(confidence, 3))

        if confidence > 0.1:
            newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 3)
            newImage = cv2.putText(newImage, label, (top_x, top_y+50), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, (0, 230, 0), 1, cv2.LINE_AA)
            
    return newImage


def average(result_list):
    if (len(result_list) == 0):
        return 0
    return int(sum(result_list.values()) / len(result_list))

def find_top_classes(result_list, result_out):
# print("Length %s\n" % range(len(result_out["detection_class_entities"])))
    for i in range(len(result_out)):
        current_class = result_out[i]["label"]
        scores = int (100 * result_out[i]["confidence"])
        if scores > min_threshold:
            if current_class not in result_list.keys(): 
                result_list[current_class] = scores
            if current_class in result_list.keys() and result_list[current_class] < scores:
                result_list[current_class] = scores

result_list = dict()
i = 0
while(True):
    if (procesing_frame_rate > frame_count):
        print("[{}] Wrong 'processing_frame_rate' value: {} < {}".format(time.strftime("%d-%m-%Y %H:%M:%S", time.localtime()), 1, 2))
        break
    # Capture frame-by-frame
    ret, frame = cap.read()
    if (i % procesing_frame_rate == 0):
        frame = np.asarray(frame)
        if (do_resize):
            frame = cv2.resize(frame, (int(height/2), int(width/2)))

        results = tfnet.return_predict(frame)

        new_frame = boxing(frame, results)
        out.write(new_frame)
        find_top_classes(result_list, results)
          
        print("[{}] Completed: {} %".format(time.strftime("%d-%m-%Y %H:%M:%S", time.localtime()),int(i / point)))
    i = i + 1
    if (cap.get(cv2.CAP_PROP_FRAME_COUNT) == i):
        break

passed_seconds = int(time.time() - start)
m, s = divmod(passed_seconds, 60)

sr.store_results(model_name, len(result_list), average(result_list), passed_seconds, video_name)

pr.sort_translate_print(result_list, model_name)

print("Total spend time: {:02d}m : {:02d}s".format(m,s))
print("=======================================")

cap.release()
out.release()