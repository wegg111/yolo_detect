'''
    配置文件，集中调整vision.py中的参数
'''
import torch

class cfg():
    sim = False          # False为接真机摄像头
    yolo_common = False  # 使用yolo_common， False为使用yolo_fast

    time_iterator = 1   # 时间计数器，指多少次循环打印debug信息
    frame_width = 640   # resolution
    frame_height = 480
    ROS1 = True          # 接入ROS1，为了方便不接ROS时测试代码

    camera_id = 5        # 使用的摄像头的序号，用于获取内参
    usb_cam = 200          # 自己电脑编号一般为0， 用camera.py检测摄像头编号

    # yolo_common模型路径
    yolo_common_model_path = './yolo_v3/models/2019_07_03_jonas_yolo/config.cfg'
    yolo_common_weight_path = './yolo_v3/models/2019_07_03_jonas_yolo/yolo_weights.weights'
    yolo_common_class_names = ['ball', 'goalpost', 'robot', 'L-Intersection', 'T-Intersection', 'X-Intersection']

    # yolo_fast加载路径（coco模型）
    # yolo_fast_data_path = '/data/coco.data'
    # yolo_fast_weight_path = '/modelzoo/coco2017-0.241078ap-model.pth'

    # yolo_fast加载路径（torso模型）
    yolo_fast_data_path = '/data/torso.data'
    yolo_fast_weight_path = '/modelzoo/torso-300-epoch-0.495576ap-model.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

