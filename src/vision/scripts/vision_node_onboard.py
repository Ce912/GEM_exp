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
from scipy.spatial.transform import Rotation as R
from vision.msg import RobotState

O_T_C = np.zeros((4, 4))
O_T_C_ext = np.zeros((4, 4))
c = 0

class POSES:
    def __init__(self):

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        #config.enable_device_from_file(bag, False)
        widht = 1280
        height = 720

        s_n_on_board = '027322072641'
        self.config.enable_device(s_n_on_board)

        self.config.enable_stream(rs.stream.color, widht, height, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, widht, height, rs.format.z16, 30)
        profile = self.pipeline.start(self.config)

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        self.device = pipeline_profile.get_device()

    def find(self):
        align_to = rs.stream.color
        align = rs.align(align_to)
        for i in range(10):
            self.pipeline.wait_for_frames()
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
        model = YOLO("/home/leon/shared_ws/angelo_ws/llm_workspace/src/vision/YOLO_fruit_best.pt")

        # YOLO results
        results = model(color_image)[0]
        detections = sv.Detections.from_ultralytics(results)

        oriented_box_annotator = sv.OrientedBoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        annotated_frame = oriented_box_annotator.annotate(
            scene=color_image, detections=detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections)

        context = ''
        # sv.plot_image(image=annotated_frame, size=(16, 16))
        for detection_idx in range(len(detections)):
            object_name = detections['class_name'][detection_idx]
            box = results[detection_idx].obb.xywhr
            color_intrin = self.color_intrin
            vdist = self.depth_frame.get_distance(box[0][0], box[0][1])
            point = rs.rs2_deproject_pixel_to_point(color_intrin, [box[0][0], box[0][1]], vdist)
            # convert local coordinates into robot base coordinates
            x_rel = point[0]
            y_rel = point[1]
            z_rel = point[2]
            theta_rel = (box[0][4].item())*180/np.pi

            rotation_matrix = O_T_C[0:3, 0:3]
            C_T_o_cmarray = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, x_rel, y_rel, z_rel, 1.0])
            C_T_o = np.array(C_T_o_cmarray, order='F').reshape((4,4), order='F')
            O_T_o = np.matmul(O_T_C, C_T_o)
            x_g = O_T_o[0, 3]*1000
            y_g = O_T_o[1, 3]*1000
            z_g = O_T_o[2, 3]*1000 - 40

            r = R.from_matrix(rotation_matrix)
            euler_angles = r.as_euler('xyz', degrees=True)
            theta_g = euler_angles[2]  - theta_rel
            # create the context
            context += str(object_name) + ',' + str(int(x_g)) + ',' + str(int(y_g)) + ',' + str(int(z_g)) + ',' + str(int(theta_g)) + ';'
        # publish context
        if (c < 1000):
            pub = rospy.Publisher('/context_from_vision_onboard', String, queue_size=2000)
            rospy.loginfo(context)
            pub.publish(context)

def callback(data):
    global O_T_C, c
    # global O_T_C_ext
    # EE_T_C (extrinsecs matrix from calibration)
    # EE_T_C_cmarray = np.array([0.002391302001311324, -0.9999469455095757, 0.010019373274216412, 0.0, 0.999947798939772, 0.00229153261675874, -0.009957322620651242, 0.0, 0.009933834619316264, 0.010042661217819144, 0.9999002269653807, 0.0, 0.027762079665860966, -0.029542410336221313, -0.038584153074454494, 1.0])
    EE_T_C_cmarray = np.array([0.022288845304241245, 0.9997492180952425, -0.002169860123366514, 0.0, -0.9996630147287806, 0.02231570617364864, 0.013261456983954008, 0.0, 0.013306553211462676, 0.00187345463492401318, 0.9999097086565906, 0.0, 0.02711385979847173, -0.03393779506264873, -0.037530544916826725, 1.0])

    
    EE_T_C = np.array(EE_T_C_cmarray, order='F').reshape((4,4), order='F')
    EE_T_C = np.array([0.0316993, 0.999491, 0.00362504, 0.0, -0.999497, 0.0317004, -0.000255266, 0.0,
                               -0.000370051, -0.00361513, 0.999993, 0.0, 0.0466949, -0.0162726, 0.0625025, 1.0]).reshape((4, 4))
    # O_T_EE
    O_T_EE_cmarray = np.array(data.O_T_EE)
    O_T_EE = np.array(O_T_EE_cmarray, order='F').reshape((4,4), order='F')
    # O_T_C
    O_T_C = np.matmul(O_T_EE, EE_T_C)
    c = c + 1

def subscriber():
    v = POSES()
    # v_ext = POSES_ext()
    # End effector pose in base frame (O_T_EE), 4x4 matrix in column-major format
    # rospy.Subscriber('/robot_state_publisher_node_1/robot_state/O_T_EE', Float64MultiArray, callback)
    rospy.Subscriber('/robot_state_publisher_node_1/robot_state', RobotState, callback)
    rate = rospy.Rate(10)
    rate.sleep()
    while not rospy.is_shutdown(): 
        v.find()
        # v_ext.find()
    rospy.spin()
  
if __name__ == '__main__': 
    rospy.init_node('vision_node', anonymous=True)  
    try: 
        subscriber() 
    except rospy.ROSInterruptException: 
        pass
