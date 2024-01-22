#!/usr/bin/env python3.8
import os
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import torch 
import numpy as np 
import time
import cv2
from pathlib import Path
from MESSAGES.msg import dets
from utils import *


'''''
This code snippet is part of an ongoing project, serving as a demonstration to showcase my coding proficiency. Here, I'm constructing a real-time vision system inegrated in a robotics node,
which is crucial for the project's functionality.
the nitty gritty details in the code are private for now, like the CUSTOMIZEDTRACKER
'''''

class AnalysisNode:
    def __init__(self, visualize=True):
        # Initialize ROS node
        rospy.init_node('predict_', anonymous=True)

        # Get the directory where the script is located
        script_dir  = os.getcwd()
        weight_path = os.path.join(script_dir, 'models/m', 'best.pt')
        tracker_path = Path(os.path.join(script_dir, 'models', '.pt'))

        # Load YOLO model
        self.model = self.yolov8_loader(weight_path)
        self.classifier_weight_path = os.path.join(script_dir, 'models', 'best.pt')

        # Tracker configuration
        self.tracker = CUSTOMIZEDTRACKER(
            model_weights=tracker_path, 
            device='cuda:0',
            fp16=False, min_hits=2, iou_threshold=0.1
        )
        self.tracker_2 = Different_CUSTOMIZEDTRACKER(
            model_weights=tracker_path,
            device='cuda:0',
            fp16=False, min_hits=1, iou_threshold=0.1
        )
        self.visualize = visualize

        # Classifier setup
        self.classifier = True
        self._dent_classifier = self.yolo_loader(self.classifier_weight_path)
        self.bridge = CvBridge()

        # ROS Subscribers and Publishers
        self.subscriber = rospy.Subscriber('/camera_topic', Image, self.callback, queue_size=10)
        self.detected_image_pub = rospy.Publisher('/published_image', Image, queue_size=10)
        self.CLASS_1 = rospy.Publisher('/class_1/bbx', dets, queue_size=10)
        self.CLASS_2 = rospy.Publisher('/class_2/bbx', dets, queue_size=10)

        rospy.loginfo("Prediction Node Started")

    def callback(self, image_msg):
        # Process incoming images
        timestamp = str(image_msg.header.stamp)
        image = self.img_reader(image_msg)[:,::-1,:]

        # Perform predictions with YOLO
        results = self.model.predict(source=image, stream=False,
                conf=0.5, iou=0.1, device=0, augment=False, save=False, save_txt=False, save_crop=False, imgsz=1280, verbose=False)

        bbx = np.array(results[0].boxes.data.cpu())
        class_1 = bbx[bbx[...,-1] == 0]
        class_2 = filter_posts(bbx[~(bbx[...,-1] == 0)], img_width=image.shape[1])

        if class_1.shape[0] > 0:
            class_1_tracks = track(class_1, self.tracker, image)
            if class_1_tracks is not None:
                class_1_publish = publish_bbx(class_1, class_1_tracks)
                self.CLASS_1.publish(class_1_publish)

                if self.classifier:
                    output = classify_crop_img(image, class_1_tracks, self._dent_classifier, imgsz=448, verbose=True)
                if self.visualize:
                    image = annotate(image, class_1_tracks, color=(255,255,255))

        if class_2.shape[0] > 0:
            class_2_tracks = track(class_2, self.tracker_2, image)
            if class_2_tracks is not None:
                class_2_publish = publish_bbx(class_2, class_2_tracks)
                self.CLASS_2.publish(class_2_publish)
                if self.visualize:
                    image = annotate(image, class_2_tracks)

        back_to_msg = self.bridge.cv2_to_imgmsg(image)
        self.detected_image_pub.publish(back_to_msg)

    def yolov8_loader(self, weights_path):
        # Load YOLO model
        model = YOLO(weights_path)
        return model

    def img_reader(self, img):
        # Convert ROS Image to OpenCV format
        cv2_img = self.bridge.imgmsg_to_cv2(img, "bgr8")
        return cv2_img

if __name__ == '__main__':
    predictor = AnalysisNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
