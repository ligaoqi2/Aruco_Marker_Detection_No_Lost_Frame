import numpy as np


# 双目相机参数
class stereoCamera(object):
    def __init__(self):
        # 左相机内参
        self.cam_matrix_left = np.array([[1785.650395, 0, 640.2794923],
                                         [0, 1786.33961, 517.4476781],
                                         [0, 0, 1]])
        # 右相机内参
        self.cam_matrix_right = np.array([[1787.026971, 0, 647.6913884],
                                          [0, 1787.636359, 516.2008247],
                                          [0, 0, 1]])

        # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = np.array([-0.095619071, 0.180614982, 0, 0, 0])
        self.distortion_r = np.array([-0.088950278, 0.158080178, 0, 0, 0])

        # 旋转矩阵
        self.R = np.array([[0.999623779, 0.012532016, -0.024397732],
                           [-0.012370657, 0.999900674, 0.006753418],
                           [0.024479943, -0.006449061, 0.99967952]])

        self.R = self.R.T  # matlab标定所以需要进行转置

        # 平移矩阵
        self.T = np.array([-126.7965076, -2.315387335, -0.378584673])

        # 焦距
        # self.focal_length = 312.0959 # 默认值，一般取立体校正后的重投影矩阵Q中的 Q[2,3]

        # 基线距离
        self.baseline = 126.7965  # 单位：mm， 为平移向量的第一个参数（取绝对值）

        # 左相机fx 或 fy
        self.cam_left_f = 1786.339

        # 图像宽度
        self.width = 1280

        # 图像高度
        self.height = 1024
