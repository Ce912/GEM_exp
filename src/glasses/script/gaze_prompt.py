#! /usr/bin/env python3
import g3pylib 
import asyncio
import logging
import os
import pandas as pd
import csv
import numpy as np
from g3pylib import connect_to_glasses 
import dotenv
import rospy
from std_msgs.msg import String
import cv2
from datetime import datetime
import queue
from threading import Lock
from ultralytics import YOLO
import supervision as sv


file_path = ""
#Connect to glasses, read images and determine focused object. Publish the info on a dedicated topic
queue_lock = Lock()
class GazePrompt:
    def __init__(self):
        rospy.init_node("gaze_prompt", anonymous=True)
        #rospy.on_shutdown()
        
        #Load environment variables
        self.g3_hostname = os.environ["G3_HOSTNAME"] = "tg03b-080203027651.local"
        dotenv.load_dotenv()

        #Set queue
        self.data_queue = queue.Queue(maxsize=1000)
        self.lock = Lock()

        #Import vision module
        path = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(path, 'best_glasses.pt')
        self.model = YOLO(full_path)

        #Setup publisher for pointer information
        self.pub_objA = rospy.Publisher('/pointerA', String, queue_size = 10)
        self.pub_objB = rospy.Publisher('/pointerB', String, queue_size = 10)

        #Publisher timer: 30 Hz
        self.timer = rospy.Timer(rospy.Duration(0.03), self.publish_pointer)

        #Initiliaze empty variables
        self.pointer = ""  #String
        self.history = []  #List
        self.old_A = ""  #String
        self.old_B = ""

        #Divide object lists in static and dynamic
        self.dynamic_obj = ["Grape", "Green-Pepper", "Corn", "Red-Block", "Green-Block", "Eggplant", "Tomato", "Banana", "Carrot"]
        self.static_obj = ["Black-Cup", "Plate", "Bowl"]
        

    async def disable_gaze_overlay(self, g3):
        #Disable the gaze overlay on the glasses
        gaze_overlay = await g3.settings.get_gaze_overlay()
        if gaze_overlay:
            success = await g3.settings.set_gaze_overlay(False)
            if success:
                rospy.loginfo("Gaze overlay successfully disabled.")
            else:
                rospy.logwarn("Failed to disable gaze overlay.")
        else:
            rospy.loginfo("Gaze overlay is already disabled.")

    async def read_frames(self):
        #Setup glasses connection without gaze overlay
        async with connect_to_glasses.with_hostname(self.g3_hostname) as g3:
            await self.disable_gaze_overlay(g3)
            async with g3.stream_rtsp(scene_camera=True, gaze=True) as streams:
                async with streams.gaze.decode() as gaze_stream, streams.scene_camera.decode() as scene_stream:
                    while not rospy.is_shutdown():
                        logging.info("Reading frames...")
                        frame, frame_timestamp = await scene_stream.get()
                        gaze, gaze_timestamp = await gaze_stream.get()
                        
                        #Align frame and gaze
                        while gaze_timestamp is None or frame_timestamp is None:
                            if frame_timestamp is None:
                                frame, frame_timestamp = await scene_stream.get()
                            if gaze_timestamp is None:
                                gaze, gaze_timestamp = await gaze_stream.get()
                        while gaze_timestamp < frame_timestamp:
                            gaze, gaze_timestamp = await gaze_stream.get()
                            while gaze_timestamp is None:
                                gaze, gaze_timestamp = await gaze_stream.get()

                        #Convert image in proper format for YOLO
                        image = frame.to_ndarray(format="bgr24")

                        #Resize image to standard resolution
                        height, width, channels = [360, 640, 3]
                        resized_image = cv2.resize(image, (width, height))

                        #Convert to np array for YOLO
                        resized_image = np.asanyarray(resized_image)
                        
                        #Read and store gaze pixel coordinates
                        if "gaze2d" in gaze:
                            gaze2d = gaze["gaze2d"]
                            gaze_xy = np.zeros(2, dtype=int)
                            x_2d = int(gaze2d[0] * width)
                            y_2d = int(gaze2d[1] * height)
                            gaze_xy[0] = x_2d
                            gaze_xy[1] = y_2d

                            img = cv2.circle(resized_image, (x_2d, y_2d), 8, (255, 0, 0),-1 )
                            
                            #File name and saving directory
                            timestamp = datetime.now().strftime("%m_%d_%Y_%H%M%S")
                            image_name = f"frame_" + timestamp + ".jpeg"
                            image_dir = "/media/idsia/Extreme Pro/GlassesDemo/"#+str(timestamp_univoco)+"/"

                            #Create folder for saving
                            if not os.path.exists(image_dir):
                                os.makedirs(image_dir)
                            image_path = os.path.join(image_dir, image_name)

                            #Save glasses frames
                            cv2.imwrite(image_path, img)
                            pointer = self.parser(resized_image, gaze_xy)
                            if pointer == "Eggplant":
                                pointer = ""
                        
                        #Parse image with YOLO and return focused obj
                        if pointer is not None: 
                        # Store pointer info in a list
                            self.history.append(pointer)
                            #Flush the queue
                            if len(self.history) >= 5:
                                rospy.sleep(1)
                                self.history =[]

                        #Repeat the loop at 10 Hz
                        rospy.sleep(0.1)

    def filter_detections(self, detections, results):
        #Filters duplicate object detections, keeping only the highest-confidence detection per object.
        # Extract class names, confidence scores, and bounding boxes
        name_list = list(detections.data['class_name'])  # Detected object names
        confidences = detections.confidence              # Confidence scores
        boxes = detections.xyxy                          # Bounding box coordinates
    
        # Dictionary to store the best detection index for each class
        best_detections = {}  # {class_name: (index, confidence)}
    
        for i, obj in enumerate(name_list):
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
    
        # Convert back to a detections object
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
    
        # Create the updated results object
        updated_results = results  
    
        updated_results = {
                'boxes': filtered_boxes_xywh,
                'confidences': np.array(filtered_confidences),
                'names': filtered_name_list
            }

        return filtered_detections, updated_results            

    #Process image and gaze through YOLO
    def parser(self, image, gaze_xy):
        results = self.model(image)[0]
        detections = sv.Detections.from_ultralytics(results)

        if detections.xyxy.size != 0:

            #Remove duplicates (only 1 obj per type)
            detections, results = self.filter_detections(detections, results)
            
            #Get detection boxes
            boxes = results['boxes']

            #Evaluate gaze-obj distance
            gaze_x = gaze_xy[0]
            gaze_y = gaze_xy[1]
            gaze_obj_xy = abs(boxes[:, :2] - np.array([gaze_x, gaze_y]))       
            gaze_obj_dist = np.linalg.norm(gaze_obj_xy, axis = 1) #evalutate norm per rows

            #Label closest obj as target obj
            pointer_idx = np.argmin(gaze_obj_dist)
            pointer = detections['class_name'][pointer_idx]

            return pointer 
    
    def publish_pointer(self, event):

        #Evaluate detection history of the last 3 seconds
        if len(self.history) >= 5:#10:

            #If len(set) is 1, all elements are identical
            if len(set(self.history[0:4])) == 1: 
                self.pointer = self.history[0]
                #rospy.loginfo('got pointer')

                if self.pointer in self.dynamic_obj: 
                    msg = String()
                    msg.data = self.pointer
                    self.pub_objA.publish(msg)
                    self.old_A = self.pointer

                elif self.pointer in self.static_obj:
                    msg = String()
                    msg.data = self.pointer
                    self.pub_objB.publish(msg)
                    self.old_B = self.pointer

                #Publish target obj info if same obj focused for 3 sec
                msg = String()
                msg.data = self.history[0]

                #Update pointer_old to keep publishing the latest info
                self.pointer_old = self.history[0]

                #Clean the history
                self.history = []
        return self.history 

    
if __name__ == '__main__':
    node = GazePrompt()
    try: 
       asyncio.run(node.read_frames())
    except rospy.ROSInterruptException:
        rospy.loginfo("Gaze stream interrupted.")









        









                             
                    

