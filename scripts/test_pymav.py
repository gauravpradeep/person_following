import argparse
import sys
import time
import cv2
import numpy as np
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils
from sort import Sort
from pymavlink import mavutil

def condition_yaw(master, heading, speed, relative=True):
    if relative:
        is_relative = 1
    else:
        is_relative = 0

    msg = master.mav.command_long_encode(
        0, 0,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,
        0,
        abs(heading),
        speed,
        heading / abs(heading) if heading != 0 else 1,
        is_relative,
        0, 0, 0)
    master.mav.send(msg)
    print(f"heading = {heading}")

def get_params(offset):
    
    '''

    P_yaw_deg and P_yaw_speed represent the corresponding P values for the respective P controllers

    '''

    P_yaw_deg = 0.08 / 3
    P_yaw_speed = 0.02

    param1 = P_yaw_deg * offset
    param2 = P_yaw_speed * offset
    param3 = offset / abs(offset)

    return param1, param2, param3

def run( model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
    lock_id = None
    tracked_bbox = []

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    tracker = Sort()  
    base_options = core.BaseOptions(
        file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
    detection_options = processor.DetectionOptions(max_results=3, score_threshold=0.3)
    options = vision.ObjectDetectorOptions(
        base_options=base_options, detection_options=detection_options)
    detector = vision.ObjectDetector.create_from_options(options)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit('ERROR: Unable to read from webcam. Please verify your webcam settings.')


        # rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = vision.TensorImage.create_from_array(image)
        detection_result = detector.detect(input_tensor)

        dets = []
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            confidence = detection.categories[0].score
            category = detection.categories[0].category_name
            if category != 'person' or confidence < 0.3:
                continue
            dets.append([bbox.origin_x, bbox.origin_y, bbox.origin_x+bbox.width, bbox.origin_y+bbox.height, confidence])

        tracked_objects = tracker.update(np.array(dets))

        max_id, max_conf = None, 0
        present = False
        best_box = []
        for i, to in enumerate(tracked_objects):
            bbox = to[:4].astype(int)
            object_id = int(to[4])
            object_conf = dets[i][4]
            if object_id == lock_id:
                present = True
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                tracked_bbox = bbox
                cv2.putText(image, f"ID: {object_id} Confidence: {object_conf} ", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                break

            if max_conf < object_conf:
                max_conf = object_conf
                max_id = object_id
                best_box = bbox
            
        if not present or lock_id is None:
            lock_id = max_id
            tracked_bbox = best_box

        if len(tracked_bbox) == 4:
            center = tracked_bbox[0] + (tracked_bbox[2] - tracked_bbox[0]) // 2
            offset = center - 320
            if abs(offset) < 25:
                param1, param2, param3 = 0, 0, 0
            else:
                param1, param2, param3 = get_params(offset)
                # condition_yaw(master, param1, param2, relative=True)  

            cv2.putText(image, f"Yaw degree: {param1:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Yaw speed: {param2:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('object_detector', image)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', help='Path of the object detection model.', required=False, default='../pretrained_weights/efficientdet_lite0.tflite')
    parser.add_argument('--cameraId', help='Id of camera.', required=False, type=int, default=0)
    parser.add_argument('--frameWidth', help='Width of frame to capture from camera.', required=False, type=int, default=640)
    parser.add_argument('--frameHeight', help='Height of frame to capture from camera.', required=False, type=int, default=480)
    parser.add_argument('--numThreads', help='Number of CPU threads to run the model.', required=False, type=int, default=-1)
    parser.add_argument('--enableEdgeTPU', help='Whether to run the model on EdgeTPU.', action='store_true', required=False, default=False)
    args = parser.parse_args()

    # Establishing connection with the drone
    # master = mavutil.mavlink_connection('udp:127.0.0.1:14550')
    # master.wait_heartbeat()

    run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight, int(args.numThreads), bool(args.enableEdgeTPU))

if __name__ == '__main__':
    main()
