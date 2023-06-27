#!/usr/bin/env python3
import cv2
import json
import time
import math
import yaml
import argparse
import numpy as np

from config import cfg

if cfg.ROS1:
    import rospy
    from std_msgs.msg import Float32MultiArray

from yolo_fast import test
from yolo_v3.detect import detector
from yolo_v3.models import load_model


class Vision():
    def __init__(self, arg):
        self.cfg = cfg            # 配置文件
        self.time_iterator = 0    # debug_get_time()的计数器

        # 防止没有信息打印报错
        self.debug = arg.debug
        self.main_loop_time = 1
        self.msg_metrix = np.array([])
        self.image = np.array([])
        self.cv_img = np.array([])

        # 加载模型
        if cfg.yolo_common == True:
            self.load_yolo_common()
        else:
            self.load_yolo_fast()
        print("=" * 42 + "\n" + " " * 10 + "vision model loaded\n" + "=" * 42)

        # 图片数据获取源
        if cfg.sim == True:
            pass            # 预期加入仿真，但是也许不加
        else:
            self.cap = cv2.VideoCapture(cfg.usb_cam)   # 打开摄像头
            self.cap.set(3, cfg.frame_width)
            self.cap.set(4, cfg.frame_height)
            # 获取相机内参
            with open('/home/nvidia/thmos_ws/src/thmos_code/thmos_vision/scripts/config/intrinsics.yaml', 'r') as f:
                result = yaml.load(f.read(), Loader=yaml.FullLoader)
                self.intrinsic = result[f'camera{self.cfg.camera_id}']
            f.close()

        # 设置ros_publisher
        if self.cfg.ROS1:
            rospy.init_node('thmos_vision')
            self.object_pub = rospy.Publisher("/vision/obj_pos", Float32MultiArray, queue_size=1)

    def load_yolo_fast(self):
        # yolo_fast识别器
        self.detector = test.Detector()
        self.classnames = self.detector.LABEL_NAMES

    def load_yolo_common(self):
        # yolo_common识别器
        self.model = load_model(self.cfg.yolo_common_model_path, self.cfg.yolo_common_weight_path)
        self.device = self.cfg.device
        self.detector = detector(self.model, self.device)
        self.classnames = self.cfg.yolo_common_class_names

    def sim_image_callback(self, image_msg):
        # 仿真的回调函数（备用）
        self.image = image_msg

    def detect(self, image):
        '''
        args：
            image(ndarray, opencv读到的图片)
        returns:
            目标矩阵：[x, y, x, y, confidence, class] 左上点 右下点
        '''
        t1 = time.time()
        img_array = self.detector.detect_image(image)
        t2 = time.time()
        self.detect_time = t2 - t1    # 时间计算器
        return img_array

    def draw_objects(self, image, array):
        '''
            标记bounding box
        args:
            image: 需要标记的图像（ndarray）
            array: 目标参数矩阵(ndarray)
        returns:
            image: 标记好的图像（ndarray）
        '''
        for i in array:
            x1, y1 = int(i[0]), int(i[1])
            x2, y2 = int(i[2]), int(i[3])

            # 在图片上标注 bounding box 、 confidence 、 distance 、class_name
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(image, '%.2f' % i[4], (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)   # 每个数据y坐标相差20
            cv2.putText(image, '%.2f' % i[6], (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)
            cv2.putText(image, self.classnames[int(i[5])], (x1, y1 - 45), 0, 0.7, (0, 255, 0), 2)
        return image

    def get_image(self):
        # 获取图像，返回ndarray
        t1 = time.time()
        if cfg.sim == False:
            success, self.cv_img = self.cap.read()  # 读摄像机图片
        t2 = time.time()
        self.get_image_time = t2 - t1    # 时间计算器
        return self.cv_img

    def debug_show_image(self):
        # debug函数，用于显示检测后标注的图像
        debug_img = self.draw_objects(self.cv_img, self.target_metrix)   # 画图
        cv2.imshow("debug_img", debug_img)    # 显示图片
        cv2.waitKey(10)    # 图像更新速率（ms）

    def debug_print(self):
        '''
        时间计算器，每time_interator次循环打印一次debug信息
        '''
        self.time_iterator += 1
        # 每n次循环打印一次
        if self.time_iterator == self.cfg.time_iterator:
            self.time_iterator = 0
            print("-" * 15 + "Vision Debug" + "-" * 15)
            # -------------------------时间信息----------------------------------
            print("[Get_img_time]    {:.4f}s".format(self.get_image_time))
            print("[Detect_time]     {:.4f}s".format(self.detect_time))
            print("[Main_loop_time]  {:.4f}s".format(self.main_loop_time))
            print("[Main_loop_FPS]   {:.4f}".format(1 / self.main_loop_time))

            # -------------------------数据信息----------------------------------
            # print(self.msg_metrix)

    def get_single_distance(self, axis, real_length, pixel1, pixel2):
        '''
        args:
            axis: char类型，'x' 或 'y'
            real_length: 真实长度
            pixel1, pixel2: float类型，两个边界的像素位置
        returns:
            distance: 与目标物的直线距离
        '''
        f = 0
        if axis == 'x':
            f = self.intrinsic[0][0]
        elif axis == 'y':
            f = self.intrinsic[1][1]
        distance = f * real_length / math.fabs(pixel2 - pixel1)  # 通过内参矩阵运算得到
        return distance

    def get_distance(self, item, x1, y1, x2, y2):
        '''
        args:
            item: string类型，传进来的bounding box里是什么东西
            x1, y1, x2, y2: float类型，bounding box左上角和右下角的像素坐标
        returns:
            distance: 与目标物的直线距离
        '''

        distance = 0
        x1, x2 = int(x1), int(x2)
        y1, y2 = int(y1), int(y2)

        if item == 'ball':
            diameter_real = 0.15
            distance = self.get_single_distance('x', diameter_real, x1, x2)

        elif item == 'goalpost':
            diameter_real = 0.12
            gray_img = cv2.cvtColor((np.array(self.cv_img)), cv2.COLOR_BGR2GRAY)
            goalpost_img = gray_img[y1:(y2 + 1), x1:(x2 + 1)]  # 裁出含有立柱的部分
            ret, bin_img = cv2.threshold(goalpost_img, 170, 255, cv2.THRESH_BINARY)
            dis_list = []
            for i in range(y2 - y1 + 1):
                try:
                    line = bin_img[i]  # 获取一行像素
                    first = -1
                    last = -1
                    for j in range(x2 - x1 + 1):  # 寻找这一行像素的左右极值位置
                        if line[j] == 255:
                            if first == -1:
                                first = j
                            else:
                                last = j - 1
                    if first != -1 and last != -1 and last - first > 1:
                        dis = self.get_single_distance('x', diameter_real, first, last)
                        dis_list.append(dis)
                except:
                    continue
            if len(dis_list) != 0:
                distance = sum(dis_list) / len(dis_list)
        else:
            distance = -1

        return distance

    def get_angle(self, x1, x2):
        '''
        args:
            x1, x2: float类型，bounding box左上角和右下角的横坐标位置
        returns:
            angle: 在水平平面内，摄像头正前方向与目标物中点之间的夹角，弧度制，右手方向为正
        '''
        u = (x1 + x2) / 2
        f = self.intrinsic[0][0]
        c = self.intrinsic[0][2]
        t = (u - c) / f
        return math.atan(t)

    def get_distance_and_angle(self):
        '''
        returns:
            target_metrix: N*[x1, y1, x2, y2, confidence, class, distance, angle]  (ndarray)
        return:
            np.array([])    (无物体)
        '''
        # 如果识别到物体则进行距离检测
        if self.objects.any():
            distances, angles = [], []
            for i in self.objects:
                distances.append(self.get_distance(self.classnames[int(i[5])], i[0], i[1], i[2], i[3]))
                angles.append(self.get_angle(i[0], i[2]))
            # N*6的objects矩阵 append N*1的distances列向量和 N*1的angles列向量
            target_metrix = np.append(self.objects, np.resize(np.array(distances, dtype=float), (len(self.objects), 1)), axis=1)
            target_metrix = np.append(target_metrix, np.resize(np.array(angles, dtype=float), (len(self.objects), 1)), axis=1)
            # N*8的矩阵target_metrix
            return target_metrix
        else:
            return np.array([])

    def pub_object_msgs(self):
        '''
        处理目标矩阵，发送消息

        将目标矩阵处理成信息矩阵： msg_metrix:  N*[class, x_mid, y_mid, distance, angle]

        returns:
            msg: 将信息矩阵压成一维向量
        '''
        msg_metrix = self.target_metrix

        if msg_metrix.any():
            # x, y 取中点
            msg_metrix[:, 0:1] = (msg_metrix[:, 0:1] + msg_metrix[:, 2:3]) / 2
            msg_metrix[:, 1:2] = (msg_metrix[:, 1:2] + msg_metrix[:, 3:4]) / 2
            # msg_metrix信息矩阵 shape： N*[class, x_mid, y_mid, distance, angle, confidence]
            msg_metrix = msg_metrix[:, [5, 0, 1, 6, 7, 4]]
            # 放入self， 可以debug
            self.msg_metrix = msg_metrix
            # 压成一维向量
            msg_metrix = np.array(msg_metrix).flatten()

        msg = Float32MultiArray(data=list(msg_metrix))
        self.object_pub.publish(msg)

    def main_loop(self):
        while not rospy.is_shutdown():
            t1 = time.time()
            try:
                self.objects = self.detect(self.get_image())    # 识别
            except:
                print("can't receive image")
                continue
            self.target_metrix = self.get_distance_and_angle()  # 获取目标矩阵

            self.debug_print()  # 打印主循环信息
            if self.debug == True:
                self.debug_show_image()    # 显示标注图片

            if cfg.ROS1:
                self.pub_object_msgs()    # 发布消息

            t2 = time.time()
            self.main_loop_time = t2 - t1   # 时间计算器



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Vision args")
    parser.add_argument("--debug", type=bool, default=False, help='show the image')  # 任意传东西都是True， 不传为False
    args = parser.parse_args()
    
    vision = Vision(args)
    vision.main_loop()




