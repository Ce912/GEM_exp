#!/usr/bin/env python 

import pyrealsense2 as rs
import numpy as np
import cv2
import copy
import math
from ultralytics import YOLO
import ipdb
import torch
from super_gradients.training import models
import supervision as sv
from IPython.display import clear_output  
import rospy 
from std_msgs.msg import String, Float64MultiArray
            
def publisher(): 
    pub = rospy.Publisher('/pub', Float64MultiArray, queue_size=2000) 
    rospy.init_node('vision_node', anonymous=True) 
    rate = rospy.Rate(10) 
    while not rospy.is_shutdown(): 
        EE_T_C_cmarray = [0.002391302001311324, -0.9999469455095757, 0.010019373274216412, 0.0, 0.999947798939772, 0.00229153261675874, -0.009957322620651242, 0.0, 0.009933834619316264, 0.010042661217819144, 0.9999002269653807, 0.0, 0.027762079665860966, -0.029542410336221313, -0.038584153074454494, 1.0]
        rospy.loginfo("Publishing")
        pub.publish(Float64MultiArray(data=EE_T_C_cmarray)) 
        rate.sleep() 

  
  
if __name__ == '__main__': 
    try: 
        publisher() 
    except rospy.ROSInterruptException: 
        pass