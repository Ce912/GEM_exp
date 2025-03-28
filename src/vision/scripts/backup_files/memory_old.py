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
from std_msgs.msg import String, Float32MultiArray
from vision.msg import RobotState

pose = np.zeros(5)

dict = {'Apple': pose,
        'Banana': pose,
        'Black-Cup': pose,
        'Bowl': pose,
        'Carrot': pose,
        'Chili': pose,
        'Coke-Can': pose,
        'Corn': pose,
        'Eggplant': pose,
        'Gear': pose,
        'Grape': pose,
        'Green-Block': pose,
        'Green-Cylinder': pose,
        'Green-Parallelepiped': pose,
        'Green-Pepper': pose,
        'Grey-Block': pose,
        'Orange': pose,
        'Pineapple': pose,
        'Plate': pose,
        'Red-Block': pose,
        'Red-Triangle': pose,
        'Rivella-Bottle': pose,
        'Tomato': pose,
        'White-Cup': pose,
        'Yellow-Ball': pose
        }

dict_ext = dict
 
pub = rospy.Publisher('/objects', String, queue_size=2000)
# pub_ext = rospy.Publisher('/objects_ext', String, queue_size=2000)

def update(data, list):
        splitted = data.split(";")
        for elements in splitted[:-1]:
            e = elements.split(",")
            pose = np.array([e[1], e[2], e[3], e[4], rospy.get_time()])
            if e[0] in list and e[3] != 0:
                list[e[0]] = pose
            # else:
                # list[e[0]] = pose
        return list

def callback(data):
    context = ''
    global dict, dict_ext
    # update context
    rospy.loginfo("Publishing objects")
    dict = update(data.data, dict)
    for key, value in dict.items(): 
        for key_ext, value_ext in dict_ext.items():
            if (key == key_ext and value[4] < value_ext[4]):
                dict[key] = dict_ext[key_ext]
        context += str(key) + ',' + str(value[0]) + ',' + str(value[1]) + ',' + str(value[2]) + ',' + str(value[3]) + ';'
    pub.publish(context)

def callback_onboard(data):
    context = ''
    global dict, dict_ext
    # update context
    rospy.loginfo("Publishing objects from onboard")
    dict = update(data.data, dict)
    for key, value in dict.items(): 
        for key_ext, value_ext in dict_ext.items():
            if (key == key_ext and value[4] < value_ext[4]):
                dict[key] = dict_ext[key_ext]
        context += str(key) + ',' + str(value[0]) + ',' + str(value[1]) + ',' + str(value[2]) + ',' + str(value[3]) + ';'
    pub.publish(context)

def listener(): 
    rospy.init_node('vision_node', anonymous=True)
    rospy.Subscriber('/context_from_vision', String, callback)
    rospy.Subscriber('/context_from_vision_onboard', String, callback_onboard) 
    rospy.spin()

if __name__ == '__main__':
    listener()