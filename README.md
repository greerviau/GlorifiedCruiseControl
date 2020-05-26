# GlorifiedCruiseControl

## Overview
Senior Capstone Project 2019-2020
A proof of concept autonomous driving system that takes a video feed from the road and predicts driving commands in real time.
This process is know as [Behavioral Cloning](https://arxiv.org/abs/1805.01954). In this case the AI system is attempting to recreate the driving behavior observed in the data.

## Installation
Git clone the repository and ```cd``` into the directory
``` 
git clone https://github.com/greerviau/GlorifiedCruiseControl.git && cd GlorifiedCruiseControl
```
Download [SCNN_lanenet_models](https://drive.google.com/open?id=1Z2HSItBayCRa3pg1CEn0S_xn8LLLwIGD) and extract the contents into ```GlorifiedCruiseControl/SCNN_lanenet```

## Preliminary
Currently this system is hardcoded to collect data from Toyota vehicles, specifically Toyota 2016 Tacoma but as long as the dbc files contain the same pin assignments for SAS, Brake, Throttle and Speed it will work. In order to adjust the code to work on different vehicles there will be a lot of adjustment that needs to be done in the ```collect_data_raw.py``` script.
## Usage 
In order to use this system you must have a Panda OBD-II CAN Bus Interface. Find that [here](https://comma.ai/shop/products/panda-obd-ii-dongle)<br/>
Plug the panda into your car and connect to your laptop using a USB cable (WIFI connection is too slow).
### Data Collection
#### Recording Data
To collect data run ```python3 collect_raw_data.py <session>``` Make sure to specify different sessions for every execution.<br/>  
Recording sessions will be saved to ```data/<session>```
#### Cleaning
If additional cleaning is required, run ```python3 clip_video.py``` While video is playing, press **q** to keyframe. Once video is done playing the program will split the video along key frames and saved to ```data_cleaned/<session>/<clips>/<splits>```<br/>   
Use this to clip out lane changes and other unwanted data.<br/>   
### Preprocessing
Once the data has been cleaned, run ```python3 preprocess_data_raw.py``` This will preprocess all of the clips in ```data_cleaned``` The subfolders of this directory must have the file structure of ```data_cleaned/<session>``` with the mp4 and csv files within.<br/>  
This will save the preprocessed data to ```data_processed``` The data will be divided into sessions but the clips will be aggregated into X.npy and Y.csv<br/>    
### Training
After preprocessing, open ```train_conv_net.py``` Make sure to specify the SAVE_PATH for the model as well as the hyperparams.<br/> 
Run ```python3 train_conv_net.py``` to train the model.
### Testing
In ```visual_interface.py``` specify the CONV_NET_MODEL directory for your saved model. Also specify if you want to record data from the test.<br/>

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
### SCNN LaneNet
https://towardsdatascience.com/tutorial-build-a-lane-detector-679fd8953132

https://github.com/cardwing/Codes-for-Lane-Detection
### OpenDBC & Cabana
https://github.com/commaai/opendbc

https://github.com/commaai/cabana
### NVIDIA Paper on predicting steering commands with a Convolutional Network
https://arxiv.org/pdf/1604.07316v1.pdf
