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
from std_msgs.msg import String 

class POSES:
    def __init__(self):

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        #config.enable_device_from_file(bag, False)
        widht = 1280
        height = 720
        self.config.enable_stream(rs.stream.color, widht, height, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, widht, height, rs.format.z16, 30)
        profile = self.pipeline.start(self.config)

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        self.device = pipeline_profile.get_device()

        self.pub = rospy.Publisher('/context_from_vision', String, queue_size=2000) 
        rospy.init_node('vision_node', anonymous=True) 
        



    def find(self):
        align_to = rs.stream.color
        align = rs.align(align_to)
        for i in range(10):
            self.pipeline.wait_for_frames()
        while True:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            self.depth_frame = depth_frame

            color_image = np.asanyarray(color_frame.get_data())
            self.color_intrin = color_frame.profile.as_video_stream_profile().intrinsics

            depth_color_frame = rs.colorizer().colorize(depth_frame)

            # Convert depth_frame to numpy array to render image in opencv
            depth_color_image = np.asanyarray(depth_color_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            color_cvt = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # YOLO prediction
            model = YOLO("/home/idsia/shared_ws/icra_ws/src/vision/YOLO_fruit_best.pt")

            # YOLO results
            results = model(color_image)[0]
            detections = sv.Detections.from_ultralytics(results)

            oriented_box_annotator = sv.OrientedBoxAnnotator()
            label_annotator = sv.LabelAnnotator()
            annotated_frame = oriented_box_annotator.annotate(
                scene=color_image, detections=detections)
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections)

            # sv.plot_image(image=annotated_frame, size=(16, 16))
            context = ''
            for detection_idx in range(len(detections)):
                object_name = detections['class_name'][detection_idx]
                box = results[detection_idx].obb.xywhr
                color_intrin = self.color_intrin
                vdist = self.depth_frame.get_distance(box[0][0], box[0][1])
                point = rs.rs2_deproject_pixel_to_point(color_intrin, [box[0][0], box[0][1]], vdist)
                context += str(object_name) + ',' + str(int(point[0]*1000)) + ',' + str(int(point[1]*1000)) + ',' + str(int(point[2]*1000)) + ',' + str(int((box[0][4].item())*180/np.pi)) + ';'
            #return context
            rospy.loginfo(context) 
            self.pub.publish(context) 

            
def publisher(): 
    v = POSES()
    rate = rospy.Rate(10) 
    while not rospy.is_shutdown(): 
        v.find()
        rate.sleep() 

  
  
if __name__ == '__main__': 
    try: 
        publisher() 
    except rospy.ROSInterruptException: 
        pass