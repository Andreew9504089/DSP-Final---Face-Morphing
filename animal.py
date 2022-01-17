# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 14:37:20 2022

@author: andre
"""

import cv2

def lion():
    img = cv2.imread("./data/0.png")
    
    img = img[125:400,130:350,:]
    img = cv2.resize(img,(200,250))
    result = img.copy()
    #left brow
    cv2.circle(img, (35,60), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (75,60), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (55,50), 3,(0, 0 ,255), -1) 
    
    #right brow
    cv2.circle(img, (120,60), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (160,60), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (140,50), 3,(0, 0 ,255), -1) 
    
    #left eye
    cv2.circle(img, (55,75), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (40,73), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (65,80), 3,(0, 0 ,255), -1) 
    
    #right eye
    cv2.circle(img, (145,75), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (135,80), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (155,78), 3,(0, 0 ,255), -1) 
    
    #nose
    cv2.circle(img, (100,155), 3,(0, 0 ,255), -1) 
    
    #mouth
    cv2.circle(img, (100,175), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (100,185), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (100,215), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (65,210), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (135,210), 3,(0, 0 ,255), -1)
    
    # edge
    cv2.circle(img, (15,75), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (20,155), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (35,195), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (55,230), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (100,235), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (140,230), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (155,195), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (170,155), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (180,75), 3,(0, 0 ,255), -1) 
    
    feature = [[35,60], [55,50], [75,60], [120,60], [140,50], [160,60],
               [40,73], [55,75], [65,80], [135,80], [145,75], [155,78],
               [100,155], [65,210], [100,175], [135,210], [100,215], [100,185],
               [15,75], [20,155], [35,195], [100,235], [155,195],
               [170,155], [180,75], [55,230], [140,230]]
    cnt = 0
    for pt in feature:
        x = int(pt[0])
        y = int(pt[1])
        text = str(cnt)
        cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX,0.3, (255, 0, 255),1, cv2.LINE_AA)
        cnt += 1    
    
    cv2.imshow("lion", img)
    return feature, result

def avatar():
    img = cv2.imread("./data/avatar.jpg")
    img = img[0:250, 300:520, :]
    img = cv2.resize(img, (200,250))
    
    result = img.copy()
    #left brow
    cv2.circle(img, (15,75), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (28,78), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (43,80), 3,(0, 0 ,255), -1) 
    
    #right brow
    cv2.circle(img, (110,70), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (130,65), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (150,60), 3,(0, 0 ,255), -1) 
    
    #left eye
    cv2.circle(img, (20,95), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (40,92), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (50,98), 3,(0, 0 ,255), -1) 
    
    #right eye
    cv2.circle(img, (120,90), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (140,82), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (155,78), 3,(0, 0 ,255), -1) 
    
    #nose
    cv2.circle(img, (80,145), 3,(0, 0 ,255), -1) 
    
    #mouth
    cv2.circle(img, (55,195), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (84,178), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (120,185), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (88,205), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (85,190), 3,(0, 0 ,255), -1)
    
    # edge
    cv2.circle(img, (10,98), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (22,150), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (33,185), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (52,220), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (94,245), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (150,210), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (173,170), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (177,127), 3,(0, 0 ,255), -1) 
    cv2.circle(img, (180,70), 3,(0, 0 ,255), -1) 
    
    feature = [[15,75], [28,78], [43,80], [110,70], [130,65], [150,60],
               [20,95], [40,92], [50,98], [120,90], [140,82], [155,78],
               [80,145], [55,195], [84,178], [120,185], [88,205], [85,190],
               [10,98], [22,150], [33,185], [94,245], [173,170],
               [177,127], [180,70], [52,220], [150,210]]
    
    cnt = 0
    for pt in feature:
        x = int(pt[0])
        y = int(pt[1])
        text = str(cnt)
        cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX,0.3, (255, 0, 255),1, cv2.LINE_AA)
        cnt += 1   
        
    cv2.imshow("avatar", img)
    
    return feature, result