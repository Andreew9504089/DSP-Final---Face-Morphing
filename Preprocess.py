# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 13:24:38 2022

@author: andrew
"""

import cv2
import dlib
import math
import numpy as np


def faceDetect(img, min_confidence=0.5):
    prototxt = "deploy.prototxt"
    caffemodel = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    modelPath = "./model/"
    
    model = cv2.dnn.readNetFromCaffe(prototxt=modelPath+prototxt, caffeModel=modelPath+caffemodel)

    (height, width) = img.shape[:2]
    
    
    input_img = cv2.dnn.blobFromImage(cv2.resize(img, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))
    
    model.setInput(input_img)
    faces = model.forward()
    
    results = []
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > min_confidence:
            box = faces[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x0, y0, x1, y1) = box.astype("int")
            results.append({"box": (x0, y0, x1 - x0, y1 - y0), "confidence": confidence})
            
    return results

def plotBoundingBox(frame):    
    bound_img = frame.copy()
    
    faces = faceDetect(bound_img, 0.6)
    
    for face in faces:
        (x,y,w,h) = face["box"]
        confidence = face["confidence"]
        
        cv2.rectangle(bound_img, (x - 10,y - 10), (x + 10 + w,y + 10 + h), (0, 0, 255), 2)
        
        text = str(round(confidence*100, 2)) + "%"
        y = y - 10 if y - 10 > 10 else y + 10
        cv2.putText(bound_img, text, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    
    return bound_img, faces

def cropOneFace(img, faces):
    highest_confidence = 0
    
    crop = np.zeros(img.shape, np.uint8)
    crop[:] = [255,255,255]
    
    for face in faces:
        confidence = face["confidence"]
        
        if confidence >= highest_confidence:
            (x,y,w,h) = face["box"]
            crop = img[y-10:y+10+h, x-10:x+10+w].copy()
    if crop.shape[1] == 0 or crop.shape[0] == 0:
        crop = np.zeros(img.shape, np.uint8)
        crop[:] = [255,255,255]

    return crop

def facialLandmarksDetection(face, predictorPath):
    predictor = dlib.shape_predictor(predictorPath)
    
    landmarks = predictor(face, dlib.rectangle(0, 0, face.shape[1]-1, face.shape[0]-1))
    
    markedFace = face.copy()
    
    for p in landmarks.parts():
        cv2.circle(markedFace, (p.x, p.y), 3,(0,250,0), -1)
        
    return markedFace, landmarks    

def rotateImage(image, angle, image_center):
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 0.8)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return result
    
def faceAlignment(face, landmarks_num=5, predictorPath = './model/shape_predictor_5_face_landmarks.dat'):
    
    markedFace, landmarks = facialLandmarksDetection(face, predictorPath)
    
    coords = np.zeros((landmarks_num,2), dtype="int")
    for i in range(0, landmarks_num):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    
    rightEyeMid = (coords[0] + coords[1]) / 2
    leftEyeMid = (coords[3] + coords[2]) / 2
    nose = coords[4]
    
    rotAngle = math.atan2(rightEyeMid[1] - leftEyeMid[1], rightEyeMid[0] - leftEyeMid[0]) * 180 / math.pi
    alignedFace = rotateImage(face, rotAngle, (int(nose[0]),int(nose[1])))
    
    boundedFrame, faces = plotBoundingBox(alignedFace)
    croppedFace = cropOneFace(alignedFace, faces)
    croppedFace = cv2.resize(croppedFace, (200, 250))
    
    return markedFace, croppedFace

def preprocess(frame):
    boundedFrame, faces = plotBoundingBox(frame)
            
    croppedFace = cropOneFace(frame, faces)
            
    markedFace, alignedFace = faceAlignment(face = croppedFace)
    
    return boundedFrame, croppedFace, markedFace, alignedFace
