# YOLOv3-Based-Face-Detection-Tracking
This is a robot project for television live. System will tracking the host's face, making the face in the middle of the screen. Main algorithm is Yolov3, trained on  WIDER FACE and tested on FDDB benchmark. Extremely fine performance!
***VIDEO***:

# Thanks to ***https://github.com/experiencor/keras-yolo3***
I used part of this project, changed some codes of it. 

# How to USE
## Environmnet Requirements:
I creadted this project on my DL desktop, its characteristics are as following:
  * |name|details|
  * |:---|:---|
  * |system|linux16.04/windows10|
  * |platform|anaconda2 or 3|
  * |GPU|GTX1080TI|
  * |CUDA|9.0|
  * |CUDNN|5.0|
  * |tensoflow-gpu|1.8.3|
  
Some other dependecies you need for DL may occur some version conflicts, just Google it!

## Here comes the code
First, download my tensorflow graph file from here:

Change the .pb file path called in the file ***loadtfpb.py***, then you can run it and see the magic! Another code ***load_tf_pb-control-robot.py*** is the file I used control my robot. You can isgore that. 

# Results
## Test with pics from the internet and myself :)
![test1](https://github.com/Chenyang-ZHU/YOLOv3-Based-Face-Detection-Tracking/blob/master/github_photo/test3_4.jpg)
![test2](https://github.com/Chenyang-ZHU/YOLOv3-Based-Face-Detection-Tracking/blob/master/github_photo/test3_34.jpg)
![test3](https://github.com/Chenyang-ZHU/YOLOv3-Based-Face-Detection-Tracking/blob/master/github_photo/test2_66.jpg)

## Compare with other detector:
![test3](https://github.com/Chenyang-ZHU/YOLOv3-Based-Face-Detection-Tracking/blob/master/github_photo/mtcnn_test.png)
![test3](https://github.com/Chenyang-ZHU/YOLOv3-Based-Face-Detection-Tracking/blob/master/github_photo/yolo3_test.png)
![test3](https://github.com/Chenyang-ZHU/YOLOv3-Based-Face-Detection-Tracking/blob/master/github_photo/roc.png)

***Have Fun!***
