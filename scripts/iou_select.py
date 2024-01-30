import argparse
import sys
import time
import cv2
import numpy as np
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

def calculate_iou(box1, box2):
    """Calculate Intersection Over Union (IoU) between two bounding boxes."""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area

def run(model: str, camera_id: int, width: int, height: int, num_threads: int, enable_edgetpu: bool) -> None:
    counter, fps = 0, 0
    start_time = time.time()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    row_size = 20
    left_margin = 24
    text_color = (0, 0, 255)
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    last_tracked_bbox = None

    base_options = core.BaseOptions(file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
    detection_options = processor.DetectionOptions(max_results=3, score_threshold=0.3)
    options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
    detector = vision.ObjectDetector.create_from_options(options)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit('ERROR: Unable to read from webcam. Please verify your webcam settings.')

        counter += 1
        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = vision.TensorImage.create_from_array(rgb_image)
        detection_result = detector.detect(input_tensor)

        dets = []
        for detection in detection_result.detections:
            if detection.categories[0].category_name != 'person':
                continue
            bbox = detection.bounding_box
            dets.append([bbox.origin_x, bbox.origin_y, bbox.origin_x + bbox.width, bbox.origin_y + bbox.height,detection.categories[0].score])

        if last_tracked_bbox is not None:
            highest_iou = 0
            best_bbox = None
            for bbox in dets:
                current_iou = calculate_iou(last_tracked_bbox, bbox)
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                if current_iou > highest_iou:
                    highest_iou = current_iou
                    best_bbox = bbox

            # Check if the best bounding box is within the frame
            frame_height, frame_width = image.shape[:2]
            if best_bbox and (best_bbox[0] < 0 or best_bbox[1] < 0 or best_bbox[2] > frame_width or best_bbox[3] > frame_height):
                last_tracked_bbox = None  # Reset tracking
            else:
                if best_bbox:
                    last_tracked_bbox = best_bbox
                    cv2.rectangle(image, (best_bbox[0], best_bbox[1]), (best_bbox[2], best_bbox[3]), (0, 255, 0), 2)

        elif dets:
            max_conf=0
            temp_best_box=[]
            for detection in dets:
                if detection[4]>max_conf:
                    max_conf=detection[4]
                    temp_best_box=detection

            last_tracked_bbox = detection  # Start tracking the first detected object


        if last_tracked_bbox:
            pass
        if counter % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count / (end_time - start_time)
            start_time = time.time()

        fps_text = 'FPS = {:.1f}'.format(fps)
        text_location = (left_margin, row_size)
        cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN, font_size, text_color, font_thickness)
        cv2.imshow('object_detector', image)

        if cv2.waitKey(1) == 27:  # Exit on ESC
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', help='Path of the object detection model.', required=False, default='efficientdet_lite0.tflite')
    parser.add_argument('--cameraId', help='Id of camera.', required=False, type=int, default=0)
    parser.add_argument('--frameWidth', help='Width of frame to capture from camera.', required=False, type=int, default=640)
    parser.add_argument('--frameHeight', help='Height of frame to capture from camera.', required=False, type=int, default=480)
    parser.add_argument('--numThreads', help='Number of CPU threads to run the model.', required=False, type=int, default=4)
    parser.add_argument('--enableEdgeTPU', help='Whether to run the model on EdgeTPU.', action='store_true', required=False, default=False)
    
    args = parser.parse_args()
    run(args.model, args.cameraId, args.frameWidth, args.frameHeight, args.numThreads, args.enableEdgeTPU)

if __name__ == '__main__':
    main()
