# GlorifiedCruiseControl V2

Version 2 is currently a work in progress

## Overview
A proof of concept autonomous driving system.<br/>
This process is know as [Behavioral Cloning](https://arxiv.org/abs/1805.01954). In this case the AI system is attempting to recreate the driving behavior observed in the data.<br/>
The the Neural Network model is adopted from NVIDIA where they use a Convolutional Neural Net to take video frames of the road and predict driving commands.<br/>

## TODO
* Refactor and objectify the codebase
* Migrate Deep Learning model to pytorch
* Take a new approach to the machine learning, design a different architecture
    * Implement Efficientnet
    * Use a Perception and planning approach instead of e2e actuator control
        * Neural Net
            * Input: camera -> Perception Model -> state vector -> Planning Model -> paths
            * Train e2e
        * Controls
            * Input: paths -> Control System -> actuator control (steer, accel/decel)
    * How to label training data for the Neural net?
        * Use SLAM to create trajectories for each video frame?
    * MPC - Model Predictive Control for actuator control from paths
* Build a better ui instead of using opencv
* Build a control interface to send messages to the car

## CONV Net Architecture
Using a convolutional architecture for predicting steering wheel commands from raw images worked the best. The architecture used was based on an [paper by NVIDIA](https://arxiv.org/pdf/1604.07316v1.pdf).

## References
### SCNN LaneNet (Not currently in use)
https://towardsdatascience.com/tutorial-build-a-lane-detector-679fd8953132<br/>
https://github.com/cardwing/Codes-for-Lane-Detection
### OpenDBC & Cabana
https://github.com/commaai/opendbc<br/>
https://github.com/commaai/cabana
### NVIDIA Paper on predicting steering commands with a Convolutional Network
https://arxiv.org/pdf/1604.07316v1.pdf
