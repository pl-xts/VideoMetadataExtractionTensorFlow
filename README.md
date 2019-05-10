# VideoMetadataExtractionTensorFlow
School project - extract contextual metadata from video using TensorFlow framework.

Used TensorFlow models:<p/>
* <a href="https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1" target="_blank">**FasterRCNN+InceptionResNetV2**</a>
* <a href="https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1" target="_blank">**SSD+MobileNetV2**</a>
* <a href="https://tfhub.dev/deepmind/i3d-kinetics-600/1" target="_blank">**Inflated 3D Convnet**</a>
* <a href="https://github.com/thtrieu/darkflow" target="_blank">**DarkFlow**</a>

Weights for YOLO are accessible <a href="https://drive.google.com/open?id=1EMLSQpShqaLLNhUIFNbWIHPoTqRjKIAo" target="_blank">**here**</a>. Put **yolo.weights** into new **./bin** folder.

Checkpoint for YOLO is available <a href="https://drive.google.com/file/d/1RiGuBkW3W_hiU7F3tNXSm5wQAq7A2zOG/view?usp=sharing" target="_blank">**here**</a>. Put all files into new **./ckpt** folder.

YOLO training instruction can be found <a href="https://github.com/thtrieu/darkflow#training-on-your-own-dataset" target="_blank">**here**</a>.
Command for YOLO training on custom data set:
```
python.exe flow.py --model ./cfg/tiny-yolo-vocCustom.cfg --load ./bin/tiny-yolo-voc.weights --train --annotation ./dataset/annotations --dataset ./dataset/images --gpu 1 --batch 20 --epoch 300 --save 4450
```
