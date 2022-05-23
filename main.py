# -*- coding:utf-8 -*-
#本程序用于大津算法的实现
import time

import cv2  #导入opencv模块
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

type = sys.getfilesystemencoding()
sys.stdout = Logger("count.txt")
print("Hello binbinlan!")     #打印“hello！”，验证模块导入成功

img = cv2.imread("lishi2.png")  #导入图片，图片放在程序所在目录
#cv2.namedWindow("imagshow", 2)   #创建一个窗口
#cv2.imshow('imagshow', img)    #显示原始图片

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #转换为灰度图

#使用局部阈值的大津算法进行图像二值化
#dst = cv2.adaptiveThreshold(gray,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,101, 1)


def on_tracebar_changed(args):
    pass
def on_buttom_changed(args):
    pass

cv2.namedWindow('Image',cv2.WINDOW_FREERATIO)
cv2.createTrackbar('thres','Image',0,255,on_tracebar_changed)
cv2.createTrackbar('erosion','Image',0,5,on_tracebar_changed)
cv2.createTrackbar('dilation','Image',0,5,on_tracebar_changed)
cv2.createTrackbar('inverse','Image',0,1,on_tracebar_changed)
while True:
    time.sleep(0.1)
    thresh = cv2.getTrackbarPos('thres','Image')
    erosion = cv2.getTrackbarPos('erosion','Image')
    dilation = cv2.getTrackbarPos('dilation', 'Image')
    inverse = cv2.getTrackbarPos('inverse', 'Image')
    k = np.ones((3,3),np.uint8)
    gray0 = cv2.erode(gray,k,iterations=erosion)
    gray1 = cv2.dilate(gray0,k,iterations=dilation)
    dst = cv2.threshold(gray1,thresh,255,cv2.THRESH_BINARY)[1]
    if inverse == 0:
        pass
    else:
        h, w = dst.shape[:2]
        imgInv = np.empty((w, h), np.uint8)
        for i in range(h):
            for j in range(w):
                dst[i][j] = 255 - dst[i][j]

    cv2.imshow('Image',dst)
    cv2.imshow('Src', img)
    k = cv2.waitKey(1)&0xFF
    if k == ord('q') :
        break


cv2.destroyAllWindows()


#全局大津算法，效果较差
#res ,dst = cv2.threshold(gray,0 ,255, cv2.THRESH_OTSU)

element = cv2.getStructuringElement(cv2.MORPH_CROSS,(1, 1))#形态学去噪
dst=cv2.morphologyEx(dst,cv2.MORPH_OPEN,element)  #开运算去噪

contours, hierarchy = cv2.findContours(dst,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  #轮廓检测函数
cv2.drawContours(img,contours,-1,(120,120,0),2)  #绘制轮廓

count=0 #砾石总数
ares_avrg=0  #砾石平均
#遍历找到的所有砾石
for cont in contours:

    ares = cv2.contourArea(cont)#计算包围性状的面积

    if ares<100:   #过滤面积小于10的形状
        continue
    count+=1    #总体计数加1
    ares_avrg+=ares

    print("{}-砾石面积:{}".format(count,ares),end="  ") #打印出每个砾石的面积

    rect = cv2.boundingRect(cont) #提取矩形坐标

    print("x:{} y:{}".format(rect[0],rect[1]))#打印坐标

    cv2.rectangle(img,rect,(0,0,0xff),1)#绘制矩形

    y=10 if rect[1]<10 else rect[1] #防止编号到图片之外

    cv2.putText(img,str(count), (rect[0], y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1) #在砾石左上角写上编号

print("砾石平均面积:{}".format(round(ares_avrg/ares,2))) #打印出每个砾石的面积

cv2.namedWindow("imagshow", cv2.WINDOW_FREERATIO)   #创建一个窗口
cv2.imshow('imagshow', img)    #显示原始图片

cv2.namedWindow("dst", cv2.WINDOW_FREERATIO)   #创建一个窗口
cv2.imshow("dst", dst)  #显示灰度图


plt.hist(gray.ravel(), 256, [0, 256]) #计算灰度直方图
plt.show()


cv2.waitKey()


#sys.stdout = Logger('/media/linux/harddisk1/lst/hanhan/log')
