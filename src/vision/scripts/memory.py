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


#This code receive the information from the vision nodes (external and onboard if 2 cameras) and publishes the pose of the obejcts on the respctive topics
pose = np.zeros(5)

#Involved objects definition
dict = {'Banana': pose,
    'Black-Cup': pose,
    'Bowl': pose,
    'Carrot': pose,
    'Corn': pose,
    'Eggplant': pose,
    'Grape': pose,
    'Green-Block': pose,
    'Green-Pepper': pose,
    'Plate': pose,
    'Red-Block': pose,
    'Tomato': pose,
    }

#Publisher
pub = rospy.Publisher('/objects', String, queue_size=2000)

#Split onboard and external camera contexts
def update(data, list, check):
        splitted = data.split(";")
        for elements in splitted[:-1]:
            e = elements.split(",")
            pose = np.array([e[1], e[2], e[3], e[4], rospy.get_time()])   
            if e[0] in list and int(e[3]) != 0 and int(e[1]) != 0 and int(e[2]) != 0 and int(e[2]) > -50 and check == 'ext':
                list[e[0]] = pose
            if e[0] in list and int(e[3]) != 0 and int(e[1]) != 0 and int(e[2]) != 0 and int(e[2]) <= -50 and check == 'onboard':
                list[e[0]] = pose
            
        return list

#Publish external camera context
def callback(data):
    context = ''
    check = 'ext'
    global dict
    # update context
    rospy.loginfo("Publishing objects external")
    dict = update(data.data, dict, check)
    for key, value in dict.items(): 
        context += str(key) + ',' + str(value[0]) + ',' + str(value[1]) + ',' + str(value[2]) + ',' + str(value[3]) + ';'
    pub.publish(context)

#Publish onboard camera context
def callback_onboard(data):
    context = ''
    check = 'onboard'
    global dict
    # update context
    rospy.loginfo("Publishing objects on_board")
    dict = update(data.data, dict, check)
    for key, value in dict.items(): 
        context += str(key) + ',' + str(value[0]) + ',' + str(value[1]) + ',' + str(value[2]) + ',' + str(value[3]) + ';'
    pub.publish(context)

#Get object to end-effector transformation (if needed)
def callback_state(data):
    global O_T_EE
    # O_T_EE
    O_T_EE_cmarray = np.array(data.O_T_EE)
    O_T_EE = np.array(O_T_EE_cmarray, order='F').reshape((4,4), order='F')

#Get information from vision node
def listener(): 
    rospy.init_node('vision_node', anonymous=True)
    rospy.Subscriber('/context_from_vision', String, callback)
    rospy.Subscriber('/context_from_vision_onboard', String, callback_onboard) 
    rospy.Subscriber('/robot_state_publisher_node_1/robot_state', RobotState, callback_state)
    rospy.spin()

if __name__ == '__main__':
    listener()