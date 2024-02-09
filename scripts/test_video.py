import argparse
import cv2
import numpy as np
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils
from sort import Sort

def get_params(offset):
    '''
    P_yaw_deg and P_yaw_speed represent the corresponding P values for the respective P controllers
    '''
    P_yaw_deg = 0.08 / 3
    P_yaw_speed = 0.015

    param1 = P_yaw_deg * offset
    param2 = P_yaw_speed * offset
    param3 = offset / abs(offset)

    return param1, param2, param3

def run(model: str, video_path: str, width: int, height: int, num_threads: int, enable_edgetpu: bool) -> None:
    lock_id = None
    tracked_bbox = []

    cap = cv2.VideoCapture(video_path)

    tracker = Sort()  
    base_options = core.BaseOptions(file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
    detection_options = processor.DetectionOptions(max_results=3, score_threshold=0.1)
    options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
    detector = vision.ObjectDetector.create_from_options(options)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        input_tensor = vision.TensorImage.create_from_array(image)
        detection_result = detector.detect(input_tensor)

        dets = []
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            confidence = detection.categories[0].score
            category = detection.categories[0].category_name
            if category != 'person' :
                continue
            dets.append([bbox.origin_x, bbox.origin_y, bbox.origin_x+bbox.width, bbox.origin_y+bbox.height, confidence])

        tracked_objects = tracker.update(np.array(dets))

        for to in tracked_objects:
            bbox = to[:4].astype(int)
            object_id = int(to[4])
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(image, f"ID: {object_id}", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('object_detector', image)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', help='Path of the object detection model.', required=False, default='../pretrained_weights/efficientdet_lite0.tflite')
    parser.add_argument('--videoPath', help='Path of the video file.', required=True)
    parser.add_argument('--frameWidth', help='Width of frame to capture from video.', required=False, type=int, default=640)
    parser.add_argument('--frameHeight', help='Height of frame to capture from video.', required=False, type=int, default=480)
    parser.add_argument('--numThreads', help='Number of CPU threads to run the model.', required=False, type=int, default=-1)
    parser.add_argument('--enableEdgeTPU', help='Whether to run the model on EdgeTPU.', action='store_true', required=False, default=False)
    args = parser.parse_args()

    run(args.model, args.videoPath, args.frameWidth, args.frameHeight, int(args.numThreads), bool(args.enableEdgeTPU))

if __name__ == '__main__':
    main()
