import sys
from rknn.api import RKNN

# env
DATASET_PATH = 'datasets/road_subset_09.txt'
model_path = 'model/yolov5.onnx'
output_path = 'model/yolov5.rknn'

do_quant = True
platform = 'rk3588'


# Create RKNN object
rknn = RKNN(verbose=False)

# Pre-process config
print('--> Config model')
rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=platform)
print('done')

# Load model
print('--> Loading model')
ret = rknn.load_onnx(model=model_path)
if ret != 0:
    print('Load model failed!')
    exit(ret)
print('done')

# Build model
print('--> Building model')
ret = rknn.build(do_quantization=do_quant, dataset=DATASET_PATH)
if ret != 0:
    print('Build model failed!')
    exit(ret)
print('done')

# Export rknn model
print('--> Export rknn model')
ret = rknn.export_rknn(output_path)
if ret != 0:
    print('Export rknn model failed!')
    exit(ret)
print('done')

# Release
rknn.release()
