# GlorifiedCruiseControl

## Overview
Senior Capstone Project 2019-2020
A perception system that detects lanes and vehicles from a video feed of the road and uses that data to train a neural network to predict my driving behavior.
This process is know as [Behavioral Cloning](https://arxiv.org/abs/1805.01954). In this case the AI system is attempting to recreate the driving behavior observed in the data.

## TODO
* Perception System
  * [x] Fix problem with shadows casted on lane lines (This is too difficult to worry about)
  * [x] Increase accuracy
  * [x] Added the use of Lanenet segmentation to improve lane line detection
* Reading Sensors
  * [x] Reading sensor data from OBDII port (UPDATE: Need to use Panda Adapter, ELM327 cannot read SAS)
  * [ ] ~~Plan B for SAS, use a gyroscopic sensor to measure angle.~~ (This doesnt work)
* Data Collection
  * [x] Start with interstate data (10 HRS) (Only was able to capture ~4, thanks COVID-19)
* Design Prediction Model
  * Possible Architectures
    * [x] Feed Forward (Perception system needs to be improved in order for this to succeed CONV Net)
    * [ ] ~~LSTM (RNN)~~ (Too computationaly expensive to run in real time)
    * [x] CONV Net on raw video frames (This works the best)
* Train Model
  * [x] Test for accuracy and avoid overfitting
* [x] Design Testing Visialization
* [x] Graphing predictions vs ground truth
* [ ] ~~Test Model In Real World~~ (Was not able to thanks again COVID-19)

## Perception System
An opencv system that detects lane lines and vehicles. 

### Lane Detection
The lane detection uses a combination of perspective transformation, color thresholding, sliding window detection and polynomial function fitting to detect the lane lines and calculate the curvature and direction of the lane.
#### SCNN Lanenet
Download the lanenet models and data [here](https://drive.google.com/open?id=1Z2HSItBayCRa3pg1CEn0S_xn8LLLwIGD)

### Vehicle Detection
#### NOTE: Did not end up using vehicle detection in conjunction with the perception system for computation limitations. Possible to implement this in the future.
A pretrained implementation of Google MobileNetSSD Network that detects vehicles from the video feed. Metrics about each vehicle are calculated such as lane position (left, right, mine) and distance to the vehicle.

## Prediction Network
A Neural Network that takes in the data collected from the perception system and predicts the driving controls (steering wheel angle, throttle, brake). **Because of limitations in the data, only steering wheel angle was predicted. This can be improved in the future.**

## CONV Net Architecture
Using a convolutional architecture for predicting steering wheel commands from raw images worked the best. The architecture used was based on an [paper by NVIDIA](https://arxiv.org/pdf/1604.07316v1.pdf).

## References
### MobileNet-SSD Object Detection
https://github.com/chuanqi305/MobileNet-SSD
### Advanced Lane Detection
https://towardsdatascience.com/teaching-cars-to-see-advanced-lane-detection-using-computer-vision-87a01de0424f
https://towardsdatascience.com/advanced-lane-detection-for-autonomous-vehicles-using-computer-vision-techniques-f229e4245e41
### OBD Python Library
https://python-obd.readthedocs.io/en/latest/
### NVIDIA Paper on predicting steering commands with a Convolutional Network
https://arxiv.org/pdf/1604.07316v1.pdf