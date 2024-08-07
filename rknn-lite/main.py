import cv2
import numpy as np
from rknnlite.api import RKNNLite
from imutils.video import FPS
import datetime
from lib.postprocess import yolov5_post_process, letterbox_reverse_box


IMG_SIZE = 640
CAM_WIDTH = 640
CAM_HEIGHT = 640
CLASSES = ("dry", "ice", "snow", "wet")

# decice tree for rk356x/rk3588
RK3588_RKNN_MODEL = 'model/yolov5.rknn'


def draw_boxes(image, boxes, confidences, class_ids, class_names, colors):
    for i, box in enumerate(boxes):
        x, y, w, h = box
        color = colors[class_ids[i]]
        label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
        # Draw the bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        # Display the label at the top of the bounding box
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        top = max(y, label_size[1])
        cv2.rectangle(image, (x, y - label_size[1]), (x + label_size[0], y), color, cv2.FILLED)
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def process_yolov5_output(output, image, class_names, confidence_threshold=0.5):
    boxes = []
    confidences = []
    class_ids = []
    height, width = image.shape[:2]
    
    # output: (1, 25200, 9)
    output = output[0]  # Remove batch dimension

    for detection in output:
        x_center, y_center, w, h, confidence, *class_scores = detection
        print(confidence)
        if confidence < confidence_threshold:
            continue
        
        class_id = np.argmax(class_scores)
        x_center *= width
        y_center *= height
        w *= width
        h *= height

        x = int(x_center - w / 2)
        y = int(y_center - h / 2)
        
        boxes.append([x, y, int(w), int(h)])
        confidences.append(float(confidence))
        class_ids.append(int(class_id))
 
    # Define colors for each class
    colors = np.random.uniform(0, 255, size=(len(class_names), 3))

    # Draw the boxes on the image
    draw_boxes(image, boxes, confidences, class_ids, class_names, colors)

if __name__ == '__main__':
    rknn_model = RK3588_RKNN_MODEL
    rknn_lite = RKNNLite()

    # load RKNN model
    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn(rknn_model)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    print('done')

#    ori_img = cv2.imread('./bus.jpg')
#    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

    # init runtime environment
    print('--> Init runtime environment')
    # run on RK356x/RK3588 with Debian OS, do not need specify target.
    ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    #Create Stream from Webcam
    vs = cv2.VideoCapture('assets/original_video.mp4')
    vs.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_SIZE)
    vs.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_SIZE)

    if not vs.isOpened():
        print('Cannot capture from camera. Exiting.')
        quit()

    # loop over the frames from the video stream
    while True:
        start = datetime.datetime.now()
        ret, frame = vs.read() 
        if not ret: break
        frame_shape = frame.shape[:2]

        # Inference
        exp_frame = cv2.resize(frame, dsize=(640,640))
        exp_frame = np.expand_dims(exp_frame, axis=0)
        #exp_frame = exp_frame.astype(np.float32) / 255.0
        outputs = rknn_lite.inference(inputs=[exp_frame])

        exp_frame = np.squeeze(exp_frame, axis=0)
        process_yolov5_output(outputs[0], exp_frame, CLASSES)

        exp_frame = cv2.resize(frame, dsize=(frame_shape[1],frame_shape[0]))

        end = datetime.datetime.now()
        total = (end-start).total_seconds()
        fps = f'FPS : {1 / total:.2f}'

        cv2.putText(exp_frame, fps, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        cv2.imshow('frame', exp_frame)
        key = cv2.waitKey(1) & 0xFF
        
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    rknn_lite.release()
    
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.release()