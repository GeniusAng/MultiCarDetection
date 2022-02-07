# MultiCarDetection

### 先放两个链接

权重文件（yolov3.weights）与配置文件（yolov3.cfg）到yolo的官网进行下载：https://pjreddie.com/darknet/yolo/

示例视频与检测视频链接：https://pan.baidu.com/s/1MkXf1cLqHLzyd7aKJjJ0zA  提取码：6666

### 项目介绍

本项目是基于视频的车辆跟踪及流量统计，是一个可跟踪路面实时车辆通行状况，并逐帧记录不同行车道车流量数目的深度学习项目。

该项目对输入的视频进行处理，主要分为以下几个步骤：
1. 使用YOLOV3模型进行目标检测
2. 使用SORT算法进行目标追踪，使用卡尔曼滤波器进行目标位置预测，并利用匈牙利算法对比目标的相似度，完成车辆目标追踪，
3. 利用虚拟线圈的思想实现车辆目标的计数，完成车流量的统计。

项目流程图如下：
![](https://github.com/GeniusAng/MultiCarDetection/blob/main/ScreenShots/%E8%BD%A6%E6%B5%81%E9%87%8F%E6%A3%80%E6%B5%8B.png)

### 效果展示
![](https://github.com/GeniusAng/MultiCarDetection/blob/main/ScreenShots/outdemo.gif)
