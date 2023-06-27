from __future__ import division

import torch
import numpy as np
import torchvision.transforms as transforms

from .utils import rescale_boxes, non_max_suppression
from .transforms import Resize, DEFAULT_TRANSFORMS


class detector():
    """
        模型类型： models.Darknet
        图片类型： ndarray
        返回:     n个物体的 [x1, y1, x2, y2, confidence, class]
        返回类型： ndarray
    """
    def __init__(self, model, device):
        self.model = model
        self.img_size = 416
        self.conf_thres = 0.5
        self.nms_thres = 0.
        self.device = device

        self.model.eval()  # Set model to evaluation mode

    def detect_image(self, image):
        image = np.array(image)
        # Configure input
        input_img = transforms.Compose([DEFAULT_TRANSFORMS,
                                        Resize(self.img_size)])((image, np.zeros((1, 5))))[0].unsqueeze(0).to(self.device)
        # Get detections
        with torch.no_grad():
            detections = self.model(input_img)
            detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)
            detections = rescale_boxes(detections[0], self.img_size, image.shape[:2])
        return detections.numpy()

