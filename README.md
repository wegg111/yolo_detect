'''THMOS vision包'''
# vision.py为主要视觉代码，通过config.py导入参数

# camera.py用于寻找外接相机序号
# ./config/intrinsics.yaml为相机标定参数文件
# 修改config文件usb_can来更改使用的摄像头标定参数

# yolo_v3是用的最初的视觉yolo版本 (RTX3080 2G GPU memoery)

########################################
    所有路径都使用相对路径，不要用绝对路径
########################################

# yolo_fast在只用cpu情况下 0.012s/次   (qzh's laptop)
# yolo_fast 占用显存1.5G
# yolo_fast 在tx2上帧率14， 占用显卡20%-50%， 占用总cpu<50%
# yolo_fast不准(coco数据集0.24（map0.5）， torso数据集0.495（map0.5）) 
# yolo_fast容易识别到很多个球

### opencv调用摄像头时间 0.03s/次， 在算力充足的情况所用时间是识别过程的3倍 ###


# 目标矩阵 (target matrix)
----------------------------------------------------------------    
    [[x1, y1, x2, y2, confidence, class, distance, angle]   
     [x1, y1, x2, y2, confidence, class, distance, angle]
     ...
     [x1, y1, x2, y2, confidence, class, distance, angle]]

    x1, y1: 目标左上角坐标
    x2, y2: 目标右下角坐标
    confidence: 置信度
    class： 目标种类序号
    distance: 目标与摄像头距离
    angle: 目标与摄像头法线所成角度

    注： 矩阵均为ndarray
----------------------------------------------------------------

# rospublic: 
    信息矩阵：msg_metrix:  N*[class, x_mid, y_mid, distance, angle, confidence]   （压为一维向量）


++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    工作流程：
        1.标定相机
        2.camera.py检测相机序号
        3.配置config.py, 包括相机序号、模型路径（按现有路径写）、开启参数
        4.vision.py
