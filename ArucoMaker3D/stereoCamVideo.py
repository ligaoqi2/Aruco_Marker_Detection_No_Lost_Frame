# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

from argparse import Namespace
import matplotlib.pyplot as plt
from PIL import Image

import makers as Maker
import camConfig as Config


def getPointXYZ(u, v, camera_matrix, d):
    X = (u - camera_matrix[0][2]) * d / camera_matrix[0][0]
    Y = (v - camera_matrix[1][2]) * d / camera_matrix[1][1]
    Z = d
    return [X, Y, Z]


def getPointRealWorldXYZ(PixelXYZ, R, T):
    X = R[0][0] * PixelXYZ[0] + R[0][1] * PixelXYZ[1] + R[0][2] * PixelXYZ[2] + T[0]
    Y = R[1][0] * PixelXYZ[0] + R[1][1] * PixelXYZ[1] + R[1][2] * PixelXYZ[2] + T[1]
    Z = R[2][0] * PixelXYZ[0] + R[2][1] * PixelXYZ[1] + R[2][2] * PixelXYZ[2] + T[2]
    return [X, Y, Z]


# # 获取畸变校正和立体校正的映射变换矩阵、重投影矩阵
def getRectifyTransform(height, width, config):
    # 读取内参和外参
    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    T = config.T

    # 计算校正变换
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion,
                                                      (width, height), R, T, alpha=0)

    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y, Q


# # 畸变校正和立体校正
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
    im_L = Image.fromarray(rectifyed_img1)  # numpy 转 image类
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)
    im_R = Image.fromarray(rectifyed_img2)  # numpy 转 image 类

    return rectifyed_img1, rectifyed_img2


# 立体校正检验----画线
def draw_line(image1, image2):
    # 建立输出图像
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]

    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2

    # 绘制等间距平行线
    line_interval = 40  # 直线间隔：50
    for k in range(height // line_interval):
        cv2.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0), thickness=2,
                 lineType=cv2.LINE_AA)

    return output


if __name__ == '__main__':

    # 加载maker
    arucoMaker = Maker.detectMaker()
    # 加载相机参数
    config = Config.stereoCamera()
    # 计算变换矩阵
    map1x, map1y, map2x, map2y, Q = getRectifyTransform(config.height, config.width, config)

    captureLeft = cv2.VideoCapture('../left.avi')
    captureRight = cv2.VideoCapture('../right.avi')

    left_flag = False
    right_flag = False
    left_last_img = []
    right_last_img = []

    count = 0

    while True:
        ret_l, frame_l = captureLeft.read()
        ret_r, frame_r = captureRight.read()

        if (not ret_l) or (not ret_r):
            break

        # 基线矫正
        iml_rectified, imr_rectified = rectifyImage(frame_l, frame_r, map1x, map1y, map2x, map2y)

        # 提取像素坐标
        pointLeft = arucoMaker.getArucoMakerCorners(iml_rectified, left_last_img, left_flag)  # 类型为： list([u,v])
        pointRight = arucoMaker.getArucoMakerCorners(imr_rectified, right_last_img, right_flag)  # 类型为： list([u,v])

        if pointLeft:
            left_flag = True
        if pointRight:
            right_flag = True
        left_last_img = [iml_rectified, pointLeft]
        right_last_img = [imr_rectified, pointRight]

        # print(pointLeft, pointRight)

        if pointLeft and pointRight:
            # 计算视差，恢复深度及3D点
            Depth = config.baseline * config.cam_left_f / (pointLeft[0] - pointRight[0])
            pointXYZ = getPointXYZ(pointLeft[0], pointLeft[1], config.cam_matrix_left, Depth)
            # print(count+1, pointXYZ)
            # RealWorldXYZ = getPointRealWorldXYZ(pointXYZ, config.R, config.T)
            # print(count+1, RealWorldXYZ)

        count += 1
