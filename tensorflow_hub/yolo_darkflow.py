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
video_name = "IMG_1048"
video_type = ".mp4"
model_name = "Yolo"
do_resize = True
cap = cv2.VideoCapture(path_to_file + video_name + video_type)
# Properties of video file
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(frame_count)
fps = cap.get(cv2.CAP_PROP_FPS )
print(fps)
# Only each [procesing_frame_rate] frame will be used for prediction
procesing_frame_rate = int(frame_count / int(frame_count/fps) / 2)
# parameter required for process completion track
point = frame_count / 100
# parameters requried in resize()
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
# parameter for enabling frame resizing
do_resize = True

options = {"model": "./cfg/yolo.cfg", 
           "load": "./bin/yolo.weights", 
           "threshold": 0.1, 
           "gpu": 1.0}

start = time.time()
tfnet = TFNet(options)

def average(result_list):
    return int(sum(result_list.values()) / len(result_list))

def find_top_classes(result_list, result_out):
# print("Length %s\n" % range(len(result_out["detection_class_entities"])))
    for i in range(len(result_out)):
        current_class = result_out[i]["label"]
        scores = int (100 * result_out[i]["confidence"])
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
        find_top_classes(result_list, results)
          
        print("[{}] Completed: {} %".format(time.strftime("%d-%m-%Y %H:%M:%S", time.localtime()),int(i / point)))
    i = i + 1
    if (cap.get(cv2.CAP_PROP_FRAME_COUNT) == i):
        break

passed_seconds = int(time.time() - start)
m, s = divmod(passed_seconds, 60)
pr.sort_translate_print(result_list, model_name)
print("Total spend time: {:02d}m : {:02d}s".format(m,s))
print("=======================================")
sr.store_results(model_name, len(result_list), average(result_list), passed_seconds, video_name)

cap.release() 