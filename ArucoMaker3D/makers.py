import cv2
import cv2.aruco as aruco
import numpy as np


class detectMaker(object):
    def __init__(self):
        # aruco_maker  init
        self.aruco_size = cv2.aruco.DICT_4X4_50
        self.aruco_dict = cv2.aruco.Dictionary_get(self.aruco_size)
        self.parameters = cv2.aruco.DetectorParameters_create()

        self.lk_params = dict(winSize=(20, 20),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # get ArucoMaker
    def getArucoMakerCorners(self, frame, last_img, flag):

        # lists of ids and the corners beloning to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame,
                                                              self.aruco_dict,
                                                              parameters=self.parameters)
        ret_point = []
        if corners:
            ret_point = corners[0][0][0].tolist()
        else:
            if last_img[0].any() and flag:
                p0 = np.float32(last_img[1]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(last_img[0], frame, p0, None, **self.lk_params)
                ret_point = p1[0][0].tolist()

        return ret_point

    # get othersMaker
    def getCircleCenter(self, frame):
        # 图像处理
        pass
        # return [x,y]
