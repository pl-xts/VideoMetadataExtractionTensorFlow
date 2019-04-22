import tensorflow as tf
import tensorflow_hub as hub

tf.logging.set_verbosity(tf.logging.ERROR)

import random
import cv2
import numpy as np
import time
from statistics import mean

from utils import prepare_results as pr
from utils import store_results as sr

# Change main paramateres
path_to_file = "./sample_video/"
video_name = "people_city_daytime"
video_type = ".mp4"
model_name = "i3d-kinetics-600"
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
do_resize = True

def average(result_list):
 return sum(result_list.values()) / len(result_list)

def load_video(path, max_frames=0, resize=(224, 224)):
  frames = []
  i = 0
  try:
    while True:
      if (procesing_frame_rate > frame_count):
        print("[{}] Wrong 'processing_frame_rate' value: {} < {}".format(time.strftime("%d-%m-%Y %H:%M:%S", time.localtime()), 1, 2))
        break
      if (i % procesing_frame_rate == 0):
      
        ret, frame = cap.read()
        if not ret:
          break
        if (do_resize):
          frame = cv2.resize(frame, resize)
       
        frame = frame[:, :, [2, 1, 0]]
        frames.append(frame)
        
        if len(frames) == max_frames:
          break

      i = i + 1
      if (cap.get(cv2.CAP_PROP_FRAME_COUNT) == i):
        break
  finally:
    cap.release()
  return np.array(frames) / 255.0


file = open("./" + model_name + "_labels.txt","r")
labels = []
for line in file.readlines():
  labels.append(line.strip())
file.close()

sample_video = load_video(path_to_file + video_name + video_type)


with tf.Graph().as_default():

  start = time.time()
  i3d = hub.Module("https://tfhub.dev/deepmind/" + model_name + "/1")
  input_placeholder = tf.placeholder(shape=(None, None, 224, 224, 3), dtype=tf.float32)
  logits = i3d(input_placeholder)
  probabilities = tf.nn.softmax(logits)
  # First add an empty dimension to the sample video as the model takes as input
  # a batch of videos.  
  
  model_input = np.expand_dims(sample_video, axis=0)
  with tf.train.MonitoredSession() as session:
    [ps] = session.run(probabilities,
                       feed_dict={input_placeholder: model_input})

passed_seconds = int(time.time() - start)
m, s = divmod(passed_seconds, 60)

print("=======================================")
names = []
scores = []
result_list = dict()
rank = 0
for i in np.argsort(ps)[::-1]:
  current_class = labels[i].capitalize()
  current_score = int(ps[i] * 100)
  if current_score > min_threshold:
    if current_class not in result_list.keys(): 
      result_list[current_class] = current_score
    if current_class in result_list.keys() and result_list[current_class] < current_score:
       result_list[current_class] = current_score


sr.store_results(model_name, len(result_list), average(result_list), passed_seconds, video_name)
pr.sort_translate_print(result_list, model_name)
print("Total spend time: {:02d}m : {:02d}s".format(m,s))
print("=======================================")