# For running inference on the TF-Hub module.
import tensorflow as tf
import tensorflow_hub as hub

# For downloading the image.
import matplotlib.pyplot as plt
import tempfile
from six import BytesIO

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

# For measuring the inference time.
import time
from datetime import datetime

import cv2
import base64

from utils import prepare_results as pr
from utils import store_results as sr

import sys
sys.path.append("./tensorflow_hub/utils")

# Change main paramateres
path_to_file = "./tensorflow_hub/sample_video/"
video_name = "IMG_1048"
video_type = ".mp4"
model_name = "Inception Resnet V2"
cap = cv2.VideoCapture(path_to_file + video_name + video_type)
# Properties of video file
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
# Only each [procesing_frame_rate] frame will be used for prediction
procesing_frame_rate = int(frame_count / int(frame_count/fps))
# parameter required for process completion track
point = frame_count / 100
# parameters requried in resize()
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 

# @param ["https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1", "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"]
module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"

def average(result_list):
 return sum(result_list.values()) / len(result_list)

def find_top_classes(result_list, result_out):
 # print("Length %s\n" % range(len(result_out["detection_class_entities"])))
  for i in range(len(result_out["detection_class_entities"])):
    current_class = result_out["detection_class_entities"][i].decode("utf-8")
    scores = int (100 * result_out["detection_scores"][i])
    if current_class not in result_list.keys(): 
      result_list[current_class] = scores
    if current_class in result_list.keys() and result_list[current_class] < scores:
       result_list[current_class] = scores
 #print("Apending: %s" % result_list)

detection_graph = tf.Graph()
with tf.Graph().as_default():
    start = time.time()
    detector = hub.Module(module_handle)

    input_placeholder = tf.placeholder(dtype=tf.uint8)

    decoded_image_float = tf.image.convert_image_dtype(
        image=input_placeholder, dtype=tf.float32)
   
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    module_input = tf.expand_dims(decoded_image_float, 0)

    result = detector(module_input, as_dict=True)

    init_ops = [tf.global_variables_initializer(), tf.tables_initializer()]

    session = tf.Session()
    session.run(init_ops)

    #result_list = dict(predicted_class="", value=0)
    result_list = dict()
    i = 0
    print("=======================================")
    while True:
        
        if (procesing_frame_rate > frame_count):
          print("[{}] Wrong 'processing_frame_rate' value: {} < {}".format(time.strftime("%d-%m-%Y %H:%M:%S", time.localtime()), 1, 2))
          break
        
        ret, image_np = cap.read()
        image_np = cv2.resize(image_np, (int(height/2), int(width/2)))
        if (i % procesing_frame_rate == 0):
          #print("Started [%s]\n" % i)
          result_out, image_out = session.run(
            [result, decoded_image_float],
            feed_dict={input_placeholder: image_np})
          find_top_classes(result_list, result_out)
          print("[{}] Completed: {} %".format(time.strftime("%d-%m-%Y %H:%M:%S", time.localtime()),int(i / point)))
        i = i + 1
        if (cap.get(cv2.CAP_PROP_FRAME_COUNT) == i):
          # print("%s\n" % result_out["detection_class_entities"])
          # print("%s\n" % result_out["detection_scores"])
          break
    
    passed_seconds = int(time.time() - start)
    m, s = divmod(passed_seconds, 60)
    
    pr.sort_translate_print(result_list, model_name)

    print("Total spend time: {:02d}m : {:02d}s".format(m,s))
    print("=======================================")
    sr.store_results(model_name, len(result_list), average(result_list), passed_seconds, video_name)
cap.release()
