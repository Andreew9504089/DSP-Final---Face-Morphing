# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 13:15:52 2022

@author: andrew
"""

import cv2
import numpy as np
#%%            

def createWeightMask(img,weight): #weight = [top, bottom, left, right] border are as fraction
    h,w = img.shape
    weightMask = np.zeros((h,w))
    
    topBorder = int(np.floor(h*weight[0]))
    bottomBorder = h - int(np.floor(h*weight[1]))
    leftBorder = int(np.floor(w*weight[2]))
    rightBorder = w - int(np.floor(w*weight[3]))
    
    topMask = np.transpose(np.tile(np.logspace(0,1,int(np.floor(h*weight[0])))/20, (rightBorder - leftBorder,1)))
    bottomMask = np.transpose(np.tile(np.logspace(1,0,int(np.floor(h*weight[1])))/20, (rightBorder - leftBorder,1)))
    leftMask = np.tile(np.logspace(0,1,int(np.floor(w*weight[2])))/20, (bottomBorder - topBorder,1))
    rightMask = np.tile(np.logspace(1,0,int(np.floor(w*weight[3])))/20, (bottomBorder - topBorder,1))    
    
    weightMask[:topBorder,leftBorder:rightBorder] = topMask
    weightMask[bottomBorder:,leftBorder:rightBorder] = bottomMask
    weightMask[topBorder:bottomBorder,:leftBorder] = leftMask
    weightMask[topBorder:bottomBorder,rightBorder:] = rightMask
    weightMask[topBorder:bottomBorder,leftBorder:rightBorder] = 1
    
    return weightMask

def faceSplitter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(gray, (7,7),0)
    energy = cv2.Sobel(imgBlur, -1,0, 1, ksize=3)
    energyBlur = cv2.medianBlur(energy, 7)
    
    ret, thresholdedEnergy = cv2.threshold(energyBlur, 0, 255, cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(thresholdedEnergy, (7,7), 0)
    ret, thresholdedEnergy = cv2.threshold(blur, 0, 255, cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
    thresholdedEnergy = cv2.medianBlur(thresholdedEnergy, 5)
    
    kernel = np.ones((5,5), np.uint8)
    thresholdedEnergy = cv2.erode(thresholdedEnergy, kernel, iterations = 1)
    thresholdedEnergy = cv2.dilate(thresholdedEnergy, kernel, iterations = 1)
    
    weight = [0.3, 0.15, 0.2, 0.2]
    weightMask = createWeightMask(thresholdedEnergy, weight)
    weightedEnergy = np.uint8(np.multiply(weightMask,thresholdedEnergy))
    
    blur = cv2.medianBlur(weightedEnergy, 5)
    ret, thresholdedEnergy = cv2.threshold(blur, 0, 255, cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(thresholdedEnergy, (5,5), 0)
    ret, featureMask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    kernel = np.ones((3,3), np.uint8)
    featureMask = cv2.erode(featureMask, kernel, iterations = 1)
    featureMask = cv2.dilate(featureMask, kernel, iterations = 1)
    
    score =[np.count_nonzero(featureMask[i,:]) for i in range(0,img.shape[0])]
    
    divisionIdx = []
    cnt = 0
    for i in range(1,len(score)-1):
        rowSum = score[i-1] + score[i] + score[i+1]
        cnt += 1
        if rowSum > 0  and score[i] == 0 and cnt >= 5:
            divisionIdx.append(i)
            cnt = 0
    
    featureMask[divisionIdx,:] = 255
    forehead = np.zeros((100,100,3))
    brow = np.zeros((100,100,3))
    eye = np.zeros((100,100,3))
    nose = np.zeros((100,100,3))
    mouth = np.zeros((100,100,3))
    
    browShift   = 0
    eyeShift    = 0
    noseShift   = 0
    mouthShift  = 0
    
    if len(divisionIdx) >= 4:
        if divisionIdx[0] - 10 >=0:
            forehead = img[:divisionIdx[0]-10, :,:].copy()
            brow = img[divisionIdx[0]-10:divisionIdx[1],:,:].copy()
            browShift = divisionIdx[0] - 10
            
        eye = img[divisionIdx[1]:divisionIdx[2]+10,:,:].copy()
        eyeShift = divisionIdx[1]
        
        for i in range(0, len(divisionIdx)):
            if divisionIdx[i] > img.shape[0]/2:
                mid = divisionIdx[i]
                if  (mid + 15)  - (divisionIdx[2]+10) > 5:
                    nose = img[divisionIdx[2]+10:mid + 10,:,:].copy()
                    noseShift = divisionIdx[2]+10
                break
        
        
        mouth = img[divisionIdx[-1]-60:,:,:].copy()
        mouthShift = divisionIdx[-1]-60
        
    split = [forehead, brow, eye, nose, mouth, featureMask]   
    shift = [browShift, eyeShift, noseShift, mouthShift]
    
    return split,shift