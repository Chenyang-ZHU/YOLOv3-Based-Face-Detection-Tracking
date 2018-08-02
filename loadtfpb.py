#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 15:38:24 2018

@author: chenyang
"""
import cv2
import tensorflow as tf
import numpy as np
import os
import argparse
import json
from utils_hhh import get_yolo_boxes, makedirs, preprocess_input
from b_box import draw_boxes,write2txt
from tqdm import tqdm




def face_detect():
    option='webcam'
    
    net_h, net_w = 416, 416 # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.5, 0.45 
    anchors = [55,69, 75,234, 133,240, 136,129, 142,363, 203,290, 228,184, 285,359, 341,260]
    def load_graph(frozen_graph_filename):
        # We load the protobuf file from the disk and parse it to retrieve the 
        # unserialized graph_def
        with open(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            
        # Then, we can use again a convenient built-in function to import a graph_def into the 
        # current default Graph
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                    graph_def, 
                    input_map=None, 
                    return_elements=None, 
                    name="prefix", 
                    op_dict=None, 
                    producer_op_list=None
                    )
        return graph

    # We use our "load_graph" function
    graph = load_graph("D:/CV/linux-project/CNN/convertor_keras_to_tensorflow-master/output_graph.pb")

    # We can verify that we can access the list of operations in the graph
# =============================================================================
#     for node in graph.as_graph_def().node:
#         print node.name
# =============================================================================

# =============================================================================
#     for op in graph.get_operations():
#         opname=op.name
#         print(opname)     # <--- printing the operations snapshot below
# =============================================================================
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions
    # We access the input and output nodes
    x = graph.get_tensor_by_name('prefix/input_1:0')
    y0 = graph.get_tensor_by_name('prefix/k2tfout_0:0')
    y1 = graph.get_tensor_by_name('prefix/k2tfout_1:0')
    y2 = graph.get_tensor_by_name('prefix/k2tfout_2:0')

    # We launch a Session
    with tf.Session(graph=graph) as sess:
        if option=='image':
            img = cv2.imread('D:/CV/linux-project/CNN/convertor_keras_to_tensorflow-master/test.jpg')
            cv2.resize(img,(640,480))
            img_h, img_w, _ = img.shape
            # compute the predicted output for test_x
            batch_input = preprocess_input(img, net_h, net_w) #416x416x3

            inputs = np.zeros((1,net_h, net_w,3),dtype='float32')
            inputs[0] = batch_input
            net_output = sess.run([y0,y1,y2],feed_dict={x:inputs}) # output=1x13x13x18

            batch_boxes = get_yolo_boxes(net_output, img_h, img_w, net_h, net_w, anchors, obj_thresh, nms_thresh)
            _,_,facecen=draw_boxes(img, batch_boxes[0], ['face'], obj_thresh) 
            print (facecen)
            cv2.imshow('image with bboxes', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            
        elif option=='webcam':
            video_reader = cv2.VideoCapture(0)
            while True:
                ret_val, img = video_reader.read()
                if ret_val == True:
                    img=cv2.resize(img,(640,480))
                    img_h, img_w, _ = img.shape
                    #print ("image size is:",img_h,",",img_w)
                    
                    # compute the predicted output for test_x
                    batch_input = preprocess_input(img, net_h, net_w) #416x416x3

                inputs = np.zeros((1,net_h, net_w,3),dtype='float32')
                inputs[0] = batch_input
                net_output = sess.run([y0,y1,y2],feed_dict={x:inputs}) # output=1x13x13x18

                batch_boxes = get_yolo_boxes(net_output, img_h, img_w, net_h, net_w, anchors, obj_thresh, nms_thresh)
                _,_,facecen=draw_boxes(img, batch_boxes[0], ['face'], obj_thresh)
                #print(facecen)
                cv2.imshow('image with bboxes', img)
                yield facecen
                if cv2.waitKey(1) == 27: 
                    break  # esc to quit
            cv2.destroyAllWindows()
            
# =============================================================================
              
            
# def return_face(face_cen):
#     if face_cen!='':
#         return face_cen
# =============================================================================
def nextyield(facey):
    nfacey=next( facey)
    return nfacey

facel=face_detect()   
         
while True: 
    facelocation=nextyield(facel)
    print(facelocation)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    