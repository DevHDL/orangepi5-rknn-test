# YOLOv5 Parameter
#OBJ_THRESH = 0.25
OBJ_THRESH = 0.5
NMS_THRESH = 0.45
IMG_SIZE = 640
# YOLOv5 Classes
CLASSES = ("dry", "ice", "snow", "wet")

# decice tree for rk356x/rk3588
DEVICE_COMPATIBLE_NODE = '/proc/device-tree/compatible'

RK3588_RKNN_MODEL = 'model/yolov5.rknn'

#Webcam dev /device/video0, /device/video1 etc.
CAM_DEV = 0
CAM_DEV2 = 2

#Capture Resolution
CAM_WIDTH = 1280
CAM_HEIGHT = 720

#Position Display
D1_WIDTH = 0
D1_HEIGHT = 200

D2_WIDTH = 640
D2_HEIGHT = 200