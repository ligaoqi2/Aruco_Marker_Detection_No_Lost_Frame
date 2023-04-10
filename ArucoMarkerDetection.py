import numpy as np
import time
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("../Peng/resultr.mp4")

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()

# Optical Flow Config
lk_params = dict(winSize=(20, 20),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

save_path = "./result.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(save_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

frame_count = 0
detect_count = 0
pre_frame = 0
pre_corners = 0
shoot_flag = 0


while True:
    ret, frame = cap.read()

    if ret:
        # 使用aruco.detectMarkers()函数可以检测到marker，返回ID和标志板的4个角点坐标
        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
        # corners -> list[0] -> array(1, 4, 2)

        if len(corners) != 0:
            shoot_flag = 1                                              # 开始拍摄
            # print("第 {} 帧检测到了Marker...".format(frame_count + 1))

            # aruco.drawDetectedMarkers(frame, corners, ids)  # 画出标志位置

            cv2.circle(frame, (int(corners[0][0][0][0]), int(corners[0][0][0][1])), color=(0, 0, 255), radius=2, thickness=-1)

            pre_frame = frame
            pre_corners = corners

            detect_count += 1
            print("0  ", corners[0][0][0])

        elif len(corners) == 0 and detect_count >= 1:
            # print("第 {} 帧没有检测到Marker, 采用光流追踪".format(frame_count))

            img0, img1 = pre_frame, frame
            p0 = np.float32(pre_corners).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)

            pre_frame = frame
            pre_corners = p1

            cv2.circle(frame, (int(p1[0][0][0]), int(p1[0][0][1])), color=(0, 255, 0), radius=2, thickness=-1)

            print("1  ", p1[0][0])

        cv2.imshow("frame", frame)

        if shoot_flag:
            out.write(frame)

        frame_count += 1

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
