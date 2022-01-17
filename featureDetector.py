# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 14:02:21 2022

@author: 
"""


import cv2
import numpy as np
from sklearn import preprocessing


def browDetector(image):
    result = image.copy().astype('uint8')
    
    img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(img.astype('uint8'), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
   
    for i in range(thresh.shape[1]):
        if thresh[0,i] == 255 and thresh[-1,i] == 255:
            thresh[:,i] = 0
             
    contours, hierarchy = cv2.findContours(thresh, 3, 2)
    
    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
    
    h,w = img.shape
    leftmost1   = (h,0)
    leftmost2   = (h,0)
    rightmost1  = (h,w)
    rightmost2  = (h,w)
    topmost1    = (0,w)
    topmost2    = (0,w)
    bottommost1 = (h,w)
    bottommost2 = (h,w)
    
    index1 = 0
    index2 = 0
    
    for i in range(len(contours)):
        cnt1 = contours[i] # one eyebrow
     
        leftmost1 = tuple(cnt1[cnt1[:,:,0].argmin()][0])
        rightmost1 = tuple(cnt1[cnt1[:,:,0].argmax()][0])
        topmost1 = tuple(cnt1[cnt1[:,:,1].argmax()][0])
        bottommost1 = tuple(cnt1[cnt1[:,:,1].argmin()][0])
        if (rightmost1[0] - leftmost1[0]) / (topmost1[1] - bottommost1[1]) > 2: # we assume that width of brow is more than double of height
            index1 = i
            break
    
    for i in range(index1 + 1, len(contours)):
        cnt2 = contours[i] # the other eyebrow
        
        leftmost2 = tuple(cnt2[cnt2[:,:,0].argmin()][0])
        rightmost2 = tuple(cnt2[cnt2[:,:,0].argmax()][0])
        bottommost2 = tuple(cnt2[cnt2[:,:,1].argmin()][0])
        topmost2 = tuple(cnt2[cnt2[:,:,1].argmax()][0])
        if (rightmost2[0] - leftmost2[0]) / (topmost2[1] - bottommost2[1]) > 2:
            index2 = i
            break    
#%%

    middlemost1_1 = np.zeros((2,1))
    middlemost1_2 = np.zeros((2,1))
    middlemost2_1 = np.zeros((2,1))
    middlemost2_2 = np.zeros((2,1))
    middlexy1 = [0,0]
    middlexy2 = [0,0]
    middlemost1_1[0] = (leftmost1[0] + rightmost1[0])/2
    middlemost1_2[0] = (leftmost1[0] + rightmost1[0])/2
    middlemost2_1[0] = (leftmost2[0] + rightmost2[0])/2
    middlemost2_2[0] = (leftmost2[0] + rightmost2[0])/2
    
#%% first eyebrow
    cnt1 = contours[index1] 
    cntclosest = np.zeros(len(cnt1[:,:,0]))
    d = {}
    for i in range(len(cnt1[:,:,0])):
        cntclosest[i] = np.round(abs(cnt1[i,:,0][0] - middlemost1_1[0][0]))

    for i, v in enumerate(cntclosest):#利用函数enumerate列出的每个元素下标i和元素v
        d[v]=i   
    
    cntclosest = sorted(cntclosest) 
    if len(cntclosest) >= 2:
        for i in range(len(cntclosest)):
            if cnt1[d[cntclosest[i]],:,1][0] - topmost1[1] < 5:
                mostsmall = cntclosest[i] #與中間x座標相差最小
                break
        for i in range(len(cntclosest)):
            if bottommost1[1] - cnt1[d[cntclosest[i]],:,1][0] < 5:
                secondsmall = cntclosest[i] #與中間x座標相差最小
                break
    
        for i in range(2,len(cntclosest)):
            if abs(cnt1[d[secondsmall],:,1][0] - cnt1[d[mostsmall],:,1][0]) < 5:
                secondsmall = cntclosest[i]

        middlemost1_1[1][0] = cnt1[d[mostsmall],:,1][0]
        middlemost1_2[1][0] = cnt1[d[secondsmall],:,1][0]   
        middlexy1[0] = int((leftmost1[0] + rightmost1[0])/2)
        middlexy1[1] = int((middlemost1_1[1][0] + middlemost1_2[1][0])/2)  
        
        middlexy1 = list(middlexy1)
    else:
        middlexy1 = (int((leftmost1[0]+rightmost1[0])/2), int((leftmost1[1]+rightmost1[1])/2))
    
#%% second eyebrow
    cnt2 = contours[index2-1]
    cntclosest = np.zeros(len(cnt2[:,:,0]))
    d = {}
    for i in range(len(cnt2[:,:,0])):
        cntclosest[i] = np.round(abs(cnt2[i,:,0][0] - middlemost2_1[0][0]))

    for i, v in enumerate(cntclosest):#利用函数enumerate列出的每个元素下标i和元素v
        d[v]=i
        
    cntclosest = sorted(cntclosest)   
    if len(cntclosest) >= 2:
        for i in range(len(cntclosest)):
            if cnt2[d[cntclosest[i]],:,1][0] - topmost2[1] < 5:
                mostsmall = cntclosest[i] #與中間x座標相差最小
                break
        for i in range(len(cntclosest)):
            if bottommost2[1] - cnt2[d[cntclosest[i]],:,1][0] < 5:
                secondsmall = cntclosest[i] #與中間x座標相差最小
                break
    
        for i in range(2,len(cntclosest)):
            if abs(cnt2[d[secondsmall],:,1][0] - cnt2[d[mostsmall],:,1][0]) < 5:
                secondsmall = cntclosest[i]   
        
        middlemost2_1[1] = cnt2[d[mostsmall],:,1][0]
        middlemost2_2[1] = cnt2[d[secondsmall],:,1][0]    
        middlexy2[0] = int((leftmost2[0] + rightmost2[0])/2)
        middlexy2[1] = int((middlemost2_1[1][0] + middlemost2_2[1][0])/2)
        

        middlexy2 = list(middlexy2)
        
    else:
        middlexy2 = (int((leftmost2[0]+rightmost2[0])/2), int((leftmost2[1]+rightmost2[1])/2))
    #%%
    cv2.circle(result, (leftmost1[0], leftmost1[1]), 3,(0, 0 ,255), -1)
    cv2.circle(result, (rightmost1[0], rightmost1[1]), 3,(0, 0, 255), -1)
    cv2.circle(result, (int(middlexy1[0]), int(middlexy1[1])), 3,(0, 0, 255), -1)
    cv2.circle(result, (leftmost2[0], leftmost2[1]), 3,(0, 0 ,255), -1)
    cv2.circle(result, (rightmost2[0], rightmost2[1]), 3,(0, 0, 255), -1)
    cv2.circle(result, (int(middlexy2[0]), int(middlexy2[1])), 3,(0, 0, 255), -1)
    if leftmost1[0] < leftmost2[0]:
        browFeature = [leftmost1, middlexy1, rightmost1, leftmost2, middlexy2, rightmost2]
    else:
        browFeature = [leftmost2, middlexy2, rightmost2, leftmost1, middlexy1, rightmost1]

    return browFeature, result

def find_right_eye_center(thresh,contours,w,h):
    for i in range(w//2, w):
        for j in range(0, h, 4):
            if thresh[j,i] == 0:
                for k in range(0, len(contours)):
                    for l in range(0, len(contours[k])):
                        for m,n in contours[k][l]:
                            if i==m and j==n:
                                return k
    return False
  
def find_left_eye_center(thresh,contours,w,h):              
    for i in range(w//2, 0, -1):
        for j in range(0, h):
            if thresh[j,i] == 0:
                for k in range(0, len(contours)):
                    for l in range(0, len(contours[k])):
                        for m,n in contours[k][l]:
                            if i==m and j==n:
                                return k
    return False
                            
def find_eye_edge(Eye, mode):
    h,w = Eye.shape
    E = [0,0]
    if mode == 0:
        for i in range(0, w):
            for j in range(h-1, -1, -1):
                if Eye[j,i] == 0:
                    #print(j, i)
                    E = [i,j]
                    break
            if E != [0,0]:
                break
    elif mode == 1:
        for i in range(w-1, -1, -1):
            for j in range(h-1, -1, -1):
                if Eye[j,i] == 0:
                    #print(j, i)
                    E = [i,j]
                    break
            if E != [0,0]:
                break
    
    return E

def eyeDetector(img):
    h,w,_ = img.shape
    result = img.copy().astype('uint8')
    #Find thresh by built-in function
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    _, thresh1 = cv2.threshold(gray.astype('uint8'), 80, 255, cv2.THRESH_BINARY_INV)
        
    #Transfer to Y Cb Cr space
    img_ycc = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2YCR_CB)

#%%
    eyemap_c = np.zeros((img.shape[0], img.shape[1]), dtype = int)
    
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            eyemap_c[i][j] = (img_ycc[i][j][2]**2 + (255-img_ycc[i][j][1])**2 + img_ycc[i][j][2]/img_ycc[i][j][1])/3
    
    #Find thresh by eyemap derived in Y Cb Cr space
    eyemap_c = (preprocessing.normalize(eyemap_c)*255)
    
    _, thresh2 = cv2.threshold(eyemap_c.astype('uint8'), 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
        
    #And two thresh together
    thresh = np.uint8(thresh1*(thresh2/255))
    contours, hierarchy = cv2.findContours(thresh, 3, 2)
    contours = list(contours)
    contours2 = []
    thresh = np.ones((h,w), dtype = np.uint8)
    
    #Eliminate small contour
    for i in range(0, len(contours)):
        if cv2.contourArea(contours[i]) > h*w/100:
            contours2.append(contours[i])

    result1 = cv2.drawContours(thresh, contours2, -1, (0,255,0), 3)
    thresh &= result1
    
    #Find eyes center
    rightEye_center = find_right_eye_center(thresh, contours2, w, h)
    leftEye_center = find_left_eye_center(thresh, contours2, w, h)
    cntR = contours2[rightEye_center]
    cntL = contours2[leftEye_center]
    
    #Set the upper and lower bound of eyes
    rRange = [int(np.min(cntR[:,:,1])),int(np.max(cntR[:,:,1])+(np.max(cntR[:,:,1])-np.min(cntR[:,:,1]))//4)]
    lRange = [int(np.min(cntL[:,:,1])),int(np.max(cntL[:,:,1])+(np.max(cntL[:,:,1])-np.min(cntL[:,:,1]))//4)]

    rE2 = [int((np.max(cntR[:,:,0])+np.min(cntR[:,:,0]))//2), int((np.max(cntR[:,:,1])+np.min(cntR[:,:,1]))//2)]
    lE2 = [int((np.max(cntL[:,:,0])+np.min(cntL[:,:,0]))//2), int((np.max(cntL[:,:,1])+np.min(cntL[:,:,1]))//2)]


#%%
    #Find thresh by horizontal energy
    edge = cv2.Sobel(gray, -1,0, 1, ksize=3)
    _, thresh3 = cv2.threshold(edge.astype('uint8'), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    rightEye = thresh3[rRange[0]:rRange[1],rE2[0]-w//8 : rE2[0]+w//8]
    leftEye = thresh3[lRange[0]:lRange[1],lE2[0]-w//8 : lE2[0]+w//8]
    
    rE1 = find_eye_edge(rightEye, 0)
    rE1[0] = rE1[0]+rE2[0]-w//8
    lE3 = find_eye_edge(leftEye, 0)
    lE3[0] = lE3[0]+lE2[0]-w//8
    rE3 = find_eye_edge(rightEye, 1)
    rE3[0] = rE3[0]+rE2[0]-w//8
    lE1 = find_eye_edge(leftEye, 1)
    lE1[0] = lE1[0]+lE2[0]-w//8
    
    a,b = lE3[0], lE3[1]
    lE3[1], lE3[0] = a,b
    a,b = lE2[0], lE2[1]
    lE2[1], lE2[0] = a,b
    a,b = lE1[0], lE1[1]
    lE1[1], lE1[0] = a,b
    a,b = rE1[0], rE1[1]
    rE1[1], rE1[0] = a,b
    a,b = rE2[0], rE2[1]
    rE2[1], rE2[0] = a,b
    a,b = rE3[0], rE3[1]
    rE3[1], rE3[0] = a,b
    
    eyeFeature = [lE3,lE2,lE1,rE1,rE2,rE3]
    
    #Create marked image
    for i in range(0, len(eyeFeature)):
        eyeFeature[i][0], eyeFeature[i][1] = eyeFeature[i][1], eyeFeature[i][0] 
        result[eyeFeature[i][1],eyeFeature[i][0]] = [255,0,0]
            
    return eyeFeature, result

def noseDetector(img):
    h,w,_ = img.shape
    result = img.copy().astype('uint8')
    
    #reduce the range of nose and find contours of dark part
    img = img[:,w*3//10:w*7//10,:]
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    _, thresh1 = cv2.threshold(gray.astype('uint8'), 80, 255, cv2.THRESH_BINARY_INV)
    contours1, hierarchy = cv2.findContours(thresh1, 3, 2)
    
    #If find nothing, continue changing threshold until finding one
    thresh = 20
    while len(contours1)==0:
        thresh = thresh+20
        _, thresh1 = cv2.threshold(gray.astype('uint8'), thresh, 255, cv2.THRESH_BINARY_INV)
        contours1, hierarchy = cv2.findContours(thresh1, 3, 2)

    #Find the middle of nose
    right = 0
    left = w
    bottom = 0
    for i in range(0, len(contours1)):
        for j in contours1[i][:,:,0]:
            if j > right:
                right = int(j)
            if j < left:
                left = int(j)
        for j in contours1[i][:,:,1]:
            if j > bottom:
                bottom = int(j)

    width = int(right-left)        
    middle = (right+left)//2
    nose = gray[bottom - h//2:bottom,int(middle-width//8):int(middle+width//8)]
    
    #Find the lightest part in the range set by contours
    n = np.unravel_index(nose.argmax(), nose.shape)
    n = [n[0]-h//2+bottom, n[1]+int(middle-width//8)]
    noseFeature = []
    noseFeature.append([n[1] + w*3//10, n[0]])
    
    for i in range(0, len(noseFeature)):
        result[noseFeature[i][1],noseFeature[i][0]] = [255,0,0]
    
    return noseFeature, result

def mouthDetector(img):
    mouthFeature = []
    
    ## Read and merge
    result = img.copy().astype('uint8')
    img_hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
    
    # Here I estimate the range of color of lips in hsv
    mask = cv2.inRange(img_hsv, (0,80,130), (7,200,250))
        
    _, thresh = cv2.threshold(mask.astype('uint8'), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
    
    
    h,w,c = result.shape
    leftmost1   = (h,0)
    leftmost2   = (h,0)
    rightmost1  = (h,w)
    rightmost2  = (h,w)
    topmost1    = (0,w)
    topmost2    = (0,w)
    bottommost1 = (h,w)
    bottommost2 = (h,w)
    center      = [0,0]
    
    index1 = 0 
    index2 = 0
    
    for i in range(1, len(contours)):
        cnt1 = contours[i] # one lip     
        leftmost1 = tuple(cnt1[cnt1[:,:,0].argmin()][0])
        rightmost1 = tuple(cnt1[cnt1[:,:,0].argmax()][0])
        topmost1 = tuple(cnt1[cnt1[:,:,1].argmin()][0])
        bottommost1 = tuple(cnt1[cnt1[:,:,1].argmax()][0])
        if (rightmost1[0] - leftmost1[0]) / (bottommost1[1] - topmost1[1]) > 1.5: # we assume that width of lip is more than 1.5 times of height
            index1 = i
            break
        
    for i in range(index1 + 1, len(contours)):
        cnt2 = contours[i] # the other lip
        leftmost2 = tuple(cnt2[cnt2[:,:,0].argmin()][0])
        rightmost2 = tuple(cnt2[cnt2[:,:,0].argmax()][0])
        topmost2 = tuple(cnt2[cnt2[:,:,1].argmin()][0])
        bottommost2 = tuple(cnt2[cnt2[:,:,1].argmax()][0])
        if (rightmost2[0] - leftmost2[0]) / (bottommost2[1] - topmost2[1]) > 1.5:
            index2 = i
            break
    
    area1 = cv2.contourArea(contours[index1])
    area2 = cv2.contourArea(contours[index2])
    
    if index2 == 0 or area1 > (area2 * 4): # here we detect total mouth (the second condition is to assure that the contour we found is a lip)
        leftmost = leftmost1
        rightmost = rightmost1
        topmost = topmost1
        bottommost = bottommost1
       
    else: # here we detect two lips
        if leftmost1[0] < leftmost2[0]:
            leftmost = leftmost1
        else:
            leftmost = leftmost2
            
        if rightmost1[0] > rightmost2[0]:
            rightmost = rightmost1
        else:
            rightmost = rightmost2
        
        if topmost1[1] < topmost2[1]:
            topmost = topmost1
        else:
            topmost = topmost2
            
        if bottommost1[1] > bottommost2[1]:
            bottommost = bottommost1
        else:
            bottommost = bottommost2
  
    
#%% Refinding exact mouth's feature points
    boxedImage = img[topmost[1]-5:bottommost[1]+5, leftmost[0]-10:rightmost[0]+10].copy()
    verShift = leftmost[0] - 10
    horShift = topmost[1] - 5
    
    gray = cv2.cvtColor(boxedImage.copy(),cv2.COLOR_BGR2GRAY)
    
    corners = cv2.goodFeaturesToTrack(gray,6,0.01,10)
    corners = np.int0(corners)
    corners = np.squeeze(corners)
    
    rightmost = corners[np.argmax(corners,axis=0)[0]]
    rightmost[0] += verShift
    rightmost[1] += horShift
    
    leftmost = corners[np.argmin(corners,axis=0)[0]]
    leftmost[0] += verShift
    leftmost[1] += horShift

#%%
    middlemost1_1 = np.zeros((2,1))
    middlemost1_2 = np.zeros((2,1))
    middlemost2_1 = np.zeros((2,1))
    middlemost2_2 = np.zeros((2,1))
    middlemost1 = np.zeros((2,1))
    middlemost2 = np.zeros((2,1))
    center[0] = (leftmost[0] + rightmost[0])/2
    center[1] = (leftmost[1] + rightmost[1])/2
    middlemost1_1[0] = (leftmost[0] + rightmost[0])/2 
    middlemost1_2[0] = (leftmost[0] + rightmost[0])/2
    middlemost2_1[0] = (leftmost[0] + rightmost[0])/2
    middlemost2_2[0] = (leftmost[0] + rightmost[0])/2
    middlemost1[0] = (leftmost[0] + rightmost[0])/2
    middlemost2[0] = (leftmost[0] + rightmost[0])/2
    
#%% here we detect total mouth
    if index2 == 0 or area1 > (area2 * 4):
        cnt1 = contours[index1] 
        cntclosest = np.zeros(len(cnt1[:,:,0]))
        d = {}
        for i in range(len(cnt1[:,:,0])):
            cntclosest[i] = np.round(abs(cnt1[i,:,0][0] - middlemost1_1[0][0]))
    
        for i, v in enumerate(cntclosest):#利用函數enumerate列出的每個元素下標i和元素v
            d[v]=i 
            
        cntclosest.sort()

        if len(cntclosest) >= 2:
            for i in range(len(cntclosest)):
                if cnt1[d[cntclosest[i]],:,1][0] - topmost[1] < 5:
                    mostsmall = cntclosest[i] #與中間x座標相差最小
                    break
                mostsmall = cntclosest[0]
            for i in range(len(cntclosest)):
                if bottommost[1] - cnt1[d[cntclosest[i]],:,1][0] < 5:
                    secondsmall = cntclosest[i] #與中間x座標相差最小
                    break
                secondsmall = cntclosest[1]

        
            for i in range(2,len(cntclosest)):
                if abs(cnt1[d[secondsmall],:,1][0] - cnt1[d[mostsmall],:,1][0]) < bottommost[1] - topmost[1] - 5: # to assure that we find is the big range of lip(lips) 
                    secondsmall = cntclosest[i]
            
            middlemost1[1][0] = cnt1[d[mostsmall],:,1][0]
            middlemost2[1][0] = cnt1[d[secondsmall],:,1][0]  

            #middlexy1[0] = int((leftmost1[0] + rightmost1[0])/2)
            #middlexy1[1] = int((middlemost1_1[1][0] + middlemost1_2[1][0])/2)  
            
            #middlexy1 = list(middlexy1)
        else:
            middlemost1 = (int((leftmost[0]+rightmost1[0])/2), int((leftmost[1]+rightmost[1])/2))
            middlemost2 = (int((leftmost[0]+rightmost1[0])/2), int((leftmost[1]+rightmost[1])/2))

#%% here we detect two lips
    else:
        cnt1 = contours[index1] 
        cntclosest = np.zeros(len(cnt1[:,:,0]))
        d = {}
        for i in range(len(cnt1[:,:,0])):
            cntclosest[i] = np.round(abs(cnt1[i,:,0][0] - middlemost1_1[0][0]))
    
        for i, v in enumerate(cntclosest):#利用函數enumerate列出的每個元素下標i和元素v
            d[v]=i   
        
        cntclosest.sort()
        if len(cntclosest) >= 2:
            for i in range(len(cntclosest)):
                if cnt1[d[cntclosest[i]],:,1][0] - topmost[1] < 5:
                    mostsmall = cntclosest[i] #與中間x座標相差最小
                    break
                else:
                    mostsmall = cntclosest[0]
            for i in range(len(cntclosest)):
                if bottommost[1] - cnt1[d[cntclosest[i]],:,1][0] < 5:
                    secondsmall = cntclosest[i] #與中間x座標相差最小
                    break
                else:
                    secondsmall = cntclosest[1]
        
            for i in range(2,len(cntclosest)):
                if abs(cnt1[d[secondsmall],:,1][0] - cnt1[d[mostsmall],:,1][0]) < bottommost1[1] - topmost1[1] - 5: # to assure that we find is the big range of lip(lips)
                    secondsmall = cntclosest[i]
    
            middlemost1_1[1][0] = cnt1[d[mostsmall],:,1][0]
            middlemost1_2[1][0] = cnt1[d[secondsmall],:,1][0]    
            
        else:
            middlemost1_1 = (int((leftmost[0]+rightmost[0])/2), int((leftmost[1]+rightmost[1])/2))
            middlemost1_2 = (int((leftmost[0]+rightmost[0])/2), int((leftmost[1]+rightmost[1])/2))

        cnt2 = contours[index2]
        cntclosest = np.zeros(len(cnt2[:,:,0]))
        d = {}
        for i in range(len(cnt2[:,:,0])):
            cntclosest[i] = np.round(abs(cnt2[i,:,0][0] - middlemost2_1[0][0]))

        for i, v in enumerate(cntclosest):#利用函数enumerate列出的每个元素下标i和元素v
            d[v]=i
           
        cntclosest.sort() 
        if len(cntclosest) >= 2:
            for i in range(len(cntclosest)):
                if cnt1[d[cntclosest[i]],:,1][0] - topmost[1] < 5:
                    mostsmall = cntclosest[i] #與中間x座標相差最小
                    break
                else:
                    mostsmall = cntclosest[0]
                    
            for i in range(len(cntclosest)):
                if bottommost[1] - cnt1[d[cntclosest[i]],:,1][0] < 5:
                    secondsmall = cntclosest[i] #與中間x座標相差最小
                    break
                else:
                    secondsmall = cntclosest[1]
                    
            for i in range(2,len(cntclosest)):
                if abs(cnt2[d[secondsmall],:,1][0] - cnt2[d[mostsmall],:,1][0]) < bottommost2[1] - topmost2[1] - 5:
                    secondsmall = cntclosest[i]   
           
            middlemost2_1[1][0] = cnt2[d[mostsmall],:,1][0]
            middlemost2_2[1][0] = cnt2[d[secondsmall],:,1][0]    
           
        else:
            middlemost2_1 = (int((leftmost[0]+rightmost[0])/2), int((leftmost[1]+rightmost[1])/2))
            middlemost2_2 = (int((leftmost[0]+rightmost[0])/2), int((leftmost[1]+rightmost[1])/2))

#%% calculate the topmost and bottommost (that is, middlemost1 and middlemost2)
        middlemost1 = (middlemost1_1[0][0], min(middlemost1_1[1][0], middlemost1_2[1][0], middlemost2_1[1][0], middlemost2_2[1][0]))        
        middlemost2 = (middlemost1_1[0][0], max(middlemost1_1[1][0], middlemost1_2[1][0], middlemost2_1[1][0], middlemost2_2[1][0]))
    
    middlemost1 = np.squeeze(middlemost1)
    middlemost2 = np.squeeze(middlemost2)
    center[0] = int((leftmost[0]+rightmost[0]+middlemost1[0]+middlemost2[0])/4)
    center[1] = int((leftmost[1]+rightmost[1]+middlemost1[1]+middlemost2[1])/4)
#%%
    cv2.circle(result, (leftmost[0], leftmost[1]), 2,(0, 0 ,255), -1)
    cv2.circle(result, (rightmost[0], rightmost[1]), 2,(0, 0 ,255), -1)
    cv2.circle(result, (int(middlemost1[0]), int(middlemost1[1])), 2,(0, 0 ,255), -1)
    cv2.circle(result, (int(middlemost2[0]), int(middlemost2[1])), 2,(0, 0 ,255), -1)
    cv2.circle(result, (int(center[0]), int(center[1])), 2,(0, 0 ,255), -1)
    
    if middlemost1[1] < middlemost2[1]:
        middlemost1[0] = int(middlemost1[0])
        mouthFeature = [leftmost, middlemost1.astype('int64'), rightmost, middlemost2.astype('int64'), center]
    else:
        mouthFeature = [leftmost, middlemost2.astype('int64'), rightmost, middlemost1.astype('int64'), center]

    return mouthFeature, result # result is the marked image

def adjustGamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def faceEdgeDetector(feature, img):
    image = img.copy()
    h,w = image.shape[0], image.shape[1]
    
    corrected = adjustGamma(image, 1.1)
    
    gray = cv2.cvtColor(corrected,cv2.COLOR_BGR2GRAY)
    
    imgBlur = cv2.GaussianBlur(gray, (3,3),1)
    
    edges = cv2.Canny(imgBlur,100,200)
    kernel = np.ones((2,2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    for i in range(feature[6][0],0,-1):
        if edges[feature[6][1],i] != 0 and feature[6][0] - i > 20:
            feature.append((i,feature[6][1]))
            break
        elif i == 1:
            feature.append((20,feature[6][1]))
            break
        
    for i in range(feature[12][0],0,-1):
        if edges[feature[12][1],i] != 0 and feature[12][0] - i > 40:
            feature.append((i,feature[12][1]))
            break
        elif i == 1:
            feature.append((20,feature[12][1]))
            break
        
    for i in range(feature[13][0],0,-1):
        if edges[feature[13][1],i] != 0 and feature[13][0] - i > 20:
            feature.append((i,feature[13][1]))
            break
        elif i == 1:
            feature.append((20,feature[13][1]))
            break
    
    for i in range(feature[16][1], h):
        if edges[i,feature[16][0]] != 0 and i - feature[16][1] > 20:
            feature.append((feature[16][0],i))
            break
        elif i == h-1:
            feature.append((feature[16][0],h - 20))
            break
    
    for i in range(feature[15][0],w):
        if edges[feature[15][1],i] != 0 and i - feature[15][0] > 20:
            feature.append((i,feature[15][1]))
            break
        elif i == w-1:
            feature.append((w - 20,feature[15][1]))
            break
        
    for i in range(feature[12][0],w):
        if edges[feature[12][1],i] != 0 and i - feature[12][0] > 40:
            feature.append((i,feature[12][1]))
            break
        elif i == w-1:
            feature.append((w - 20,feature[12][1]))
            break
        
    for i in range(feature[11][0],w):
        if edges[feature[11][1],i] != 0 and i - feature[11][0] > 20:
            feature.append((i,feature[11][1]))
            break
        elif i == w-1:
            feature.append((w - 20,feature[11][1]))
            break
        
    for i in range(0, feature[13][0]):
        x = feature[13][0] - i
        y = feature[13][1] + (i + 1)
        if y < h:
            if edges[y,x] != 0 and feature[13][0] - x > 20 and y - feature[13][1] > 20:
                break
            else:
                x = int((feature[20][0]+feature[21][0])/2)
                y = int((feature[20][1]+feature[21][1])/2) 
                break
    feature.append((x,y))
        
    for i in range(0, w - feature[15][0]):
        x = feature[15][0] + i
        y = feature[15][1] + (i + 1)
        if y < h:
            if edges[y,x] != 0 and  x - feature[15][0] > 20 and y - feature[15][1] > 20:
                break
            else:
                x = int((feature[21][0]+feature[22][0])/2)
                y = int((feature[21][1]+feature[22][1])/2)
    feature.append((x,y))


    return feature