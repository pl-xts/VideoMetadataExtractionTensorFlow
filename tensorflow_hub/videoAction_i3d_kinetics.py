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

import sys
sys.path.append("./tensorflow_hub/utils")

def load_video(path, max_frames=0, resize=(224, 224)):
  cap = cv2.VideoCapture(path)
  frames = []
  try:
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      frame = cv2.resize(frame, resize)
      frame = frame[:, :, [2, 1, 0]]
      frames.append(frame)
      
      if len(frames) == max_frames:
        break
  finally:
    cap.release()
  return np.array(frames) / 255.0


file = open("./tensorflow_hub/i3d-kinetics-400_labels.txt","r")
labels = []
for line in file.readlines():
  labels.append(line.strip())
file.close()

sample_video = load_video("./tensorflow_hub/sample_video/smaller.mp4")
video_type = "default"
model_name = "i3d-kinetics-400"
# First add an empty dimension to the sample video as the model takes as input
# a batch of videos.
model_input = np.expand_dims(sample_video, axis=0)

with tf.Graph().as_default():
  start = time.time()
  i3d = hub.Module("https://tfhub.dev/deepmind/i3d-kinetics-400/1")
  input_placeholder = tf.placeholder(shape=(None, None, 224, 224, 3), dtype=tf.float32)
  logits = i3d(input_placeholder)
  probabilities = tf.nn.softmax(logits)
  with tf.train.MonitoredSession() as session:
    [ps] = session.run(probabilities,
                       feed_dict={input_placeholder: model_input})

passed_seconds = int(time.time() - start)
m, s = divmod(passed_seconds, 60)

print("Top 5 actions:")
names = []
scores = []
rank = 0
for i in np.argsort(ps)[::-1][:5]:
  names.append(labels[i].capitalize())
  scores.append(int(ps[i] * 100))

pr.sort_translate_print(dict(zip(names, scores)), model_name)

print("Total spend time: {:02d}m : {:02d}s".format(m,s))
print("=======================================")
sr.store_results(model_name, len(names), mean(scores), passed_seconds, video_type)