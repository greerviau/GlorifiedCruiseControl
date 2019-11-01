# GlorifiedCruiseControl

## Overview
Senior Capstone Project 2019-2020
A perception system that detects lanes and vehicles from a video feed of the road and uses that data to train a neural network to predict my driving behavior.

## TODO
* Perception System
  * Fix problem with shadows casted on lane lines
  * Increase accuracy
* Reading Sensors
  * Reading sensor data from OBDII port
  * Plan B for SAS, use a gyroscopic sensor to measure angle.
* Data Collection
  * Start with interstate data (10 HRS)
* Design Prediction Model
  * Possible Architectures
    * Feed Forward
    * LSTM (RNN)
* Train Model
  * Test for accuracy and avoid overfitting
* Design Testing Visialization
* Graphing predictions vs ground truth
* Test Model In Real World
  * Drive with system in real time

## Perception System
An opencv system that detects lane lines and vehicles. 

### Lane Detection
The lane detection uses a combination of perspective transformation, color thresholding, sliding window detection and polynomial function fitting to detect the lane lines and calculate the curvature and direction of the lane.

### Vehicle Detection
A pretrained implementation of Google MobileNetSSD Network that detects vehicles from the video feed. Metrics about each vehicle are calculated such as lane position (left, right, mine) and distance to the vehicle.

## Prediction Network
A Neural Network that takes in the data collected from the perception system and predicts the driving controls (steering wheel angle, throttle, brake)

## Refferences

