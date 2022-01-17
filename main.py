#!/usr/bin/env python
# coding: utf-8
"""
Created on Sat Jan  8 13:15:52 2022

@author: andrew
"""
#%%

import cv2
import numpy as np
from urllib.request import urlretrieve
from os.path import exists, join
from Preprocess import preprocess
from faceSplitter import faceSplitter
from featureDetector import browDetector, eyeDetector, noseDetector, mouthDetector, faceEdgeDetector
from animal import lion, avatar

#%%
def checkFile(fileName, savePath, retrieveURL):
    if not exists(join(savePath, fileName)):
        urlretrieve(retrieveURL+fileName, savePath+fileName)
    return

# extract facial feature with our own designed method, either by conventional cv method like haar feature or interest point, etc
# or by training our own model to compare with others' methods
def myFacialFeature(face):
    
    # forehead, brow, eye, nose, mouth, featureMask
    faceSplit, shift = faceSplitter(face)
    
    browFeature, browMarked = browDetector(faceSplit[1].astype('float32'))
    eyeFeature, eyeMarked = eyeDetector(faceSplit[2])
    noseFeature, noseMarked = noseDetector(faceSplit[3].astype('float32'))
    mouthFeature, mouthMarked = mouthDetector(faceSplit[4])
    
    for i in range(len(browFeature)):
        feat = list(browFeature[i])
        feat[1] += shift[0]
        browFeature[i] = tuple(feat)
        
    for i in range(len(eyeFeature)):
        feat = list(eyeFeature[i])
        feat[1] += shift[1]
        eyeFeature[i] = tuple(feat)
        
    for i in range(len(noseFeature)):
        feat = list(noseFeature[i])
        feat[1] += shift[2]
        noseFeature[i] = tuple(feat) 
        
    for i in range(len(mouthFeature)):
        feat = list(mouthFeature[i])
        feat[1] += shift[3]
        mouthFeature[i] = tuple(feat)
        
    feature = [browFeature, eyeFeature, noseFeature, mouthFeature]        
    
    feat=[]
    for i in range(len(feature)):
        feat += feature[i]
    
    feature = faceEdgeDetector(feat, face)
    
    return feature

def plotFeaturePoints(feature, img):
    image = img.copy()

    for pt in feature:
        x = int(pt[0])
        y = int(pt[1])
        cv2.circle(image, (x,y), 3,(0, 0 ,255), -1)  
        
    cnt = 0
    for pt in feature:
        x = int(pt[0])
        y = int(pt[1])
        text = str(cnt)
        cv2.putText(image, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX,0.3, (255, 0, 255),1, cv2.LINE_AA)
        cnt += 1    
        
    return image

def triangularCrop(image, pts):
    mask = np.zeros(image.shape, dtype = np.uint8)
    img = image.copy()
    
    points = np.array([pts])
    cv2.drawContours(mask, [points], -1, (1,1,1), -1, cv2.LINE_AA)
    tri = np.multiply(mask, img)
            
    return  tri, mask

def triangularTransform(pts1, pts2, img):
    h,w = img.shape[:2]
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    
    M1 = cv2.getAffineTransform(np.array(pts1), np.array(pts2))
    transformed = cv2.warpAffine(img.copy(), M1, (w,h), flags=2)
    
    return transformed

# transform two image with the partitions to match the feature points
def myAffineTransformation(face1, feature1, feature2):
    pt32 = (0, 0)
    pt31 = (0, int(face1.shape[0]/2))
    pt30 = (0, face1.shape[0])
    pt29 = (face1.shape[1], face1.shape[0])
    pt28 = (face1.shape[1], int(face1.shape[0]/2))
    pt27 = (face1.shape[1], 0)
    pt33 = (int(face1.shape[1]/2), 0)
    pt34 = (int(face1.shape[1]/2), face1.shape[0])
    
    feature1.extend([pt27, pt28, pt29, pt30, pt31, pt32, pt33, pt34])
    feature2.extend([pt27, pt28, pt29, pt30, pt31, pt32, pt33, pt34])
    
    index = [(0,6,18), (0,6,7), (0,1,7), (1,2,7), (2,7,8), (2,3,8), (3,8,9), (3,9,10), (3,4,10), (4,5,10),
           (5,10,11), (5,11,24), (6,18,19), (6,12,19), (6,7,12), (7,8,12), (8,9,12), (9,10,12), (10,11,12),
           (11,12,23), (11,23,24), (13,19,20), (12,13,19), (12,13,14), (12,14,15), (12,15,23), (15,22,23), 
           (13,16,21), (13,16,17), (13,14,17), (14,15,17), (15,16,17), (15,16,21), (13,20,25), (13,21,25),
           (0,1,32), (0,18,32), (18,31,32), (18,19,31), (19,20,31), (20,30,31), (21,30,34), (21,26,29),
           (21,29,34), (22,28,29), (22,23,28), (23,24,28), (24,27,28), (5,24,27), (4,5,27), (22,26,29),
           (4,27,33), (3,4,33), (2,3,33), (1,2,33), (1,32,33), (15,21,26), (15,22,26), (20,25,30), (21,25,30)]
    
    fullImage = np.zeros((face1.shape), dtype = np.int8)
    for idx in index:
        pts1 = [list(feature1[idx[0]]), list(feature1[idx[1]]), list(feature1[idx[2]])]
        pts2 = [list(feature2[idx[0]]), list(feature2[idx[1]]), list(feature2[idx[2]])]

        tri, mask = triangularCrop(face1, pts1)        
        transformed = triangularTransform(pts1, pts2, tri)
        fullImage = cv2.add(fullImage, transformed, dtype = cv2.CV_8UC3)
            
    hsv = cv2.cvtColor(fullImage.copy(), cv2.COLOR_BGR2HSV)
    _,thresh = cv2.threshold(hsv[:,:,2], 250, 255, cv2.THRESH_BINARY_INV)
    for y in range(5, thresh.shape[0] - 5):
        for x in range(5, thresh.shape[1] - 5):
            if thresh[y,x] == 0:
                sumR = []
                sumG = []
                sumB = []
                for i in range(-5,5):
                    for j in range(-5,5):
                        if thresh[y+i,x+j] == 255:
                            sumR.append(fullImage[y+i,x+j, 0])
                            sumG.append(fullImage[y+i,x+j, 1])
                            sumB.append(fullImage[y+i,x+j, 2])

                fullImage[y,x,0] = np.median(sumR)
                fullImage[y,x,1] = np.median(sumG)
                fullImage[y,x,2] = np.median(sumB)
    
    return fullImage

# apply image blending on two transformed face as inputs
def myImageBlending(face1, face2):
    # alpha blending
    mask = np.ones((face1.shape))
    mask[:,:,:] = 3
    blur2 = cv2.GaussianBlur(face2, (7,7),0)
    blur1 = cv2.GaussianBlur(face1, (17,17),0) // 4
    
    face = cv2.subtract(face1, blur1, dtype=cv2.CV_8UC3)

    result = cv2.addWeighted(np.float32(face), 0.5, blur2, 0.6, 1, dtype=cv2.CV_8UC3)
    
    result = cv2.medianBlur(result,5)
    return result
#%%

def twoImages(srcPath, refPath):
    src = cv2.imread(srcPath)
    ref = cv2.imread(refPath)
    
    boundedFrame1, croppedFace1, markedFace1, alignedFace1 = preprocess(src)
    boundedFrame2, croppedFace2, markedFace2, alignedFace2 = preprocess(ref)
    
    # split = [forehead, brow, eye, nose, mouth, featureMask]
    facialLandmarks1 = myFacialFeature(alignedFace1)
    facialLandmarks2  = myFacialFeature(alignedFace2)
    
    srcShow = alignedFace1.copy()
    refShow = alignedFace2.copy()
    
    srcLabeled = plotFeaturePoints(facialLandmarks1, srcShow)
    refLabeled= plotFeaturePoints(facialLandmarks2, refShow)
    
    transformedSrc = myAffineTransformation(srcShow, facialLandmarks1, facialLandmarks2)
    
    fuseResult = myImageBlending(transformedSrc, refShow)
    directResult =  myImageBlending(srcShow, refShow)
    

    cv2.imshow("fused", fuseResult)
    cv2.imshow("transformedSRC", transformedSrc)
    cv2.imshow("src", srcLabeled)
    cv2.imshow("ref", refLabeled)
    cv2.imshow("direct", directResult)

    cv2.waitKey(0)

def singleImage(srcPath, refType):
    src = cv2.imread(srcPath)
    
    boundedFrame1, croppedFace1, markedFace1, alignedFace1 = preprocess(src)
    facialLandmarks1 = myFacialFeature(alignedFace1)
    if refType == 'lion':
        facialLandmarks2, refShow = lion()
    elif refType == 'avatar':
        facialLandmarks2, refShow = avatar()
        
    srcShow = alignedFace1.copy()
    srcLabeled = plotFeaturePoints(facialLandmarks1, srcShow)
    transformedSrc = myAffineTransformation(srcShow, facialLandmarks1, facialLandmarks2)
    
    fuseResult = myImageBlending(transformedSrc, refShow)
    directResult =  myImageBlending(srcShow, refShow)
    
    cv2.imshow("fused", fuseResult)
    cv2.imshow("transformedSRC", transformedSrc)
    cv2.imshow("src", srcLabeled)
    cv2.imshow("direct", directResult)
    cv2.waitKey(0)
#%%
prototxt = "deploy.prototxt"
caffemodel = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
modelPath = "./model/"

checkFile(prototxt, modelPath, "https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector")
checkFile(caffemodel, modelPath, "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/")

if not exists("./model/shape_predictor_5_face_landmarks.dat"):
    urlretrieve("https://github.com/davisking/dlib-models/blob/master/shape_predictor_5_face_landmarks.dat.bz2", "./model/shape_predictor_5_face_landmarks.dat.bz2")


#%%
mode = 2
if mode == 0:
    sourcePath = "./data/leehong.jpg"
    referencePath = "./data/chriswu.jpg"
    
    twoImages(sourcePath, referencePath)

elif mode == 1:
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")
    
    
    print("Hit space to capture photo, press esc when finished")
    
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)
    
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "./data/user.png"
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
    
    cam.release()
    
    cv2.destroyAllWindows()
    
    sourcePath = "./data/user.png"
    refType = 'avatar'
    
    singleImage(sourcePath, refType)
    
elif mode == 2:
    
    sourcePath = "./data/chriswu.jpg"
    refType = 'avatar'
    
    singleImage(sourcePath,refType)
    
else:
    print("Please enter mode 1, 2, or 3")






