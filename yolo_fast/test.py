import os
import cv2
import time
import sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
import utils
import torch
import model.detector
from vision import cfg


class Detector():
    def __init__(self):

        data_path = cfg.yolo_fast_data_path
        weights_path = cfg.yolo_fast_weight_path

        data_path = os.path.dirname(__file__) + data_path
        weights_path = os.path.dirname(__file__) + weights_path

        self.cfg = utils.load_datafile(data_path)

        assert os.path.exists(data_path), "请指定正确的数据路径"
        assert os.path.exists(weights_path), "请指定正确的模型路径"

        # 模型加载
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.detector.Detector(self.cfg["classes"], self.cfg["anchor_num"], True).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))

        # sets the module in eval node
        self.model.eval()

        # 加载label names
        self.LABEL_NAMES = []
        with open(os.path.dirname(__file__) + self.cfg["names"], 'r') as f:
            for line in f.readlines():
                self.LABEL_NAMES.append(line.strip())
        f.close()

    def detect_image(self, img):
        # 图像预处理
        ori_img = img
        res_img = cv2.resize(img, (self.cfg["width"], self.cfg["height"]), interpolation=cv2.INTER_LINEAR)
        img = res_img.reshape(1, self.cfg["height"], self.cfg["width"], 3)
        img = torch.from_numpy(img.transpose(0, 3, 1, 2))
        img = img.to(self.device).float() / 255.0

        # 模型推理
        preds = self.model(img)

        # 特征图后处理
        output = utils.handel_preds(preds, self.cfg, self.device)
        # output_boxes是列表里面存一个二维tensor
        output_boxes = utils.non_max_suppression(output, conf_thres=0.3, iou_thres=0.4)

        h, w, _ = ori_img.shape
        scale_h, scale_w = h / self.cfg["height"], w / self.cfg["width"]

        # 坐标转换
        boxs_array = output_boxes[0].numpy()
        boxs_array[:, 0] *= scale_w
        boxs_array[:, 1] *= scale_h
        boxs_array[:, 2] *= scale_w
        boxs_array[:, 3] *= scale_h

        return boxs_array


