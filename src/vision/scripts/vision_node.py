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

#This code provides the 3d poses of the obejcts by parsing the camera images through YOLO
#Initialize extrinsic matrix
O_T_C = np.zeros((4, 4))
O_T_C_ext = np.zeros((4, 4))


class POSES:
    #Camera pipeline
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        widht = 848
        height = 480

        s_n_external = '218622276769' #RS D405
        self.config.enable_device(s_n_external)

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

        # YOLO path 
        model = YOLO("./best.pt")

        # YOLO results
        results = model(color_image)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections, results = self.filter_detections(detections, results, color_image)
        
        oriented_box_annotator = sv.OrAientedBoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        annotated_frame = oriented_box_annotator.annotate(
            scene=color_image, detections=detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections)

        context = ''
        # sv.plot_image(image=annotated_frame, size=(16, 16))
        for detection_idx in range(len(detections)):
            object_name = detections['class_name'][detection_idx]

            #New YOLO
            box = results['boxes'][detection_idx]

            color_intrin = self.color_intrin

            #New YOLO
            vdist = self.depth_frame.get_distance(box[0], box[1])
            point = rs.rs2_deproject_pixel_to_point(color_intrin, [box[0], box[1]], vdist)


            # convert local coordinates into robot base coordinates
            x_rel = point[0]
            y_rel = point[1]
            z_rel = point[2]

            theta_rel = (box[4].item()) * 180 / np.pi #CHECK!!

            rotation_matrix = O_T_C[0:3, 0:3]
            C_T_o_cmarray = np.array(
                [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, x_rel, y_rel, z_rel, 1.0])
            C_T_o = np.array(C_T_o_cmarray, order='F').reshape((4, 4), order='F')
            O_T_o = np.matmul(O_T_C, C_T_o)
            x_g = O_T_o[0, 3] * 1000
            y_g = O_T_o[1, 3] * 1000
            z_g = O_T_o[2, 3] * 1000 - 40  # - 15

            r = R.from_matrix(rotation_matrix)
            euler_angles = r.as_euler('xyz', degrees=True)
            theta_g = euler_angles[2] - theta_rel
            # create the context
            context += str(object_name) + ',' + str(int(x_g)) + ',' + str(int(y_g)) + ',' + str(int(z_g)) + ',' + str(
                int(theta_g)) + ';'
        # publish context
        pub = rospy.Publisher('/context_from_vision', String, queue_size=2000)
        rospy.loginfo(context)
        pub.publish(context)
    def filter_detections(self, detections, results, image):
            
            #Filters duplicate object detections, keeping only the highest-confidence detection per object.
            #(sv.Detections): The original detections object.
            #Returns:sv.Detections: A new detections object with duplicates removed.
        
            # Extract class names, confidence scores, and bounding boxes
            name_list = list(detections.data['class_name'])  # Detected object names
            confidences = detections.confidence              # Confidence scores
            boxes = detections.xyxy                          # Bounding box coordinates
        
            # Dictionary to store the best detection index for each class
            best_detections = {}  # {class_name: (index, confidence)}
        
            for i, obj in enumerate(name_list):
                box = results[i].boxes.xywh
                if obj not in best_detections or confidences[i] > best_detections[obj][1]:
                    best_detections[obj] = (i, confidences[i])  # Store best index and confidence
        
            # Keep only the best detections for duplicate objects
            filtered_name_list = []
            filtered_confidences = []
            filtered_boxes = []
        
            for i, obj in enumerate(name_list):
                if (obj in best_detections) and (best_detections[obj][0] == i):  # Keep only the highest confidence detection
                    filtered_name_list.append(obj)
                    filtered_confidences.append(confidences[i])
                    filtered_boxes.append(boxes[i])
            filtered_boxes = np.array(filtered_boxes)
        
            # Convert back to a Detections object
            filtered_detections = sv.Detections(
                xyxy=np.array(filtered_boxes),
                confidence=np.array(filtered_confidences),
                class_id=np.array([list(best_detections.keys()).index(obj) for obj in filtered_name_list]),
                data={'class_name': np.array(filtered_name_list)}
            )

            # Convert xyxy (x_min, y_min, x_max, y_max) to xywh (x_center, y_center, width, height)
            filtered_boxes_xywh = np.zeros_like(filtered_boxes)
            filtered_boxes_xywh[:, 0] = (filtered_boxes[:, 0] + filtered_boxes[:, 2]) / 2  # x_center
            filtered_boxes_xywh[:, 1] = (filtered_boxes[:, 1] + filtered_boxes[:, 3]) / 2  # y_center
            filtered_boxes_xywh[:, 2] = filtered_boxes[:, 2] - filtered_boxes[:, 0]  # width
            filtered_boxes_xywh[:, 3] = filtered_boxes[:, 3] - filtered_boxes[:, 1]  # height

            #Uncomment the following lines to get angle estimation
            boxes_r = []
            x_min = filtered_boxes[:, 0]
            y_min = filtered_boxes[:, 2]
            x_max = filtered_boxes[:, 1]
            y_max = filtered_boxes[:, 3]


            for i in range(len(x_min)):
                crop = image[int(y_min[i]):int(y_max[i]), int(x_min[i]):int(x_max[i])]  # Crop the object

                # Convert to grayscale and apply Canny edge detection
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)

                # Find contours
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(contours) == 0:
                    return None  # No object found

                # Find the minimum area rectangle (OBB)
                rect = cv2.minAreaRect(contours[0])  # (center_x, center_y), (width, height), angle

                rect = np.asarray(rect)
                angle = rect[3]
                boxes_r.append(angle)
            
            boxes_r = np.array(boxes_r) #Check dimenstion 
            filtered_boxes_xywhr = np.concatenate((filtered_boxes_xywh, boxes_r), axis = 1)
            
            updated_results = {
                    'boxes': filtered_boxes_xywhr,  #filtered_boxes_xywh 
                    'confidences': np.array(filtered_confidences),
                    'names': filtered_name_list
                }

            return filtered_detections, updated_results

def callback(data):
    global O_T_C
    # global O_T_C_ext
    # EE_T_C (extrinsecs matrix from calibration)
    # EE_T_C_cmarray = np.array([0.002391302001311324, -0.9999469455095757, 0.010019373274216412, 0.0, 0.999947798939772, 0.00229153261675874, -0.009957322620651242, 0.0, 0.009933834619316264, 0.010042661217819144, 0.9999002269653807, 0.0, 0.027762079665860966, -0.029542410336221313, -0.038584153074454494, 1.0])
    # TODO: cambare parametri

    #End effector to camera matrix
    EE_T_C_cmarray = np.array(
        [0.022288845304241245, 0.9997492180952425, -0.002169860123366514, 0.0, -0.9996630147287806, 0.02231570617364864,
         0.013261456983954008, 0.0, 0.013306553211462676, 0.00187345463492401318, 0.9999097086565906, 0.0,
         0.02711385979847173, -0.03393779506264873, -0.037530544916826725, 1.0])
    EE_T_C = np.array(EE_T_C_cmarray, order='F').reshape((4, 4), order='F')
    #Extrinsic matrix
    O_T_C =  np.array([[ 0.00414945,  0.78021702, -0.62549515 , 0.5992],
                    [ 0.99932014 , 0.01967936,  0.03117658, -0.0806],
                    [ 0.03663384, -0.62519927 ,-0.77960492 , 0.4187],
                    [ 0,        0,          0,          1      ]])



def subscriber():
    v = POSES()
    # End effector pose in base frame (O_T_EE), 4x4 matrix in column-major format
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
