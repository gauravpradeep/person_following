##width=640 height=480
import argparse
import sys
import time

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils
from sort import Sort
import numpy as np

lock_id=None
tracked_bbox=[]
P_yaw_deg=0.08
P_yaw_speed=0.05

def get_params(offset):
  global P_yaw_speed
  global P_yaw_deg

  param1=P_yaw_deg*offset
  param2=P_yaw_speed*offset
  param3=offset/abs(offset)

  return param1,param2,param3


def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
    # Variables to calculate FPS
    global lock_id
    global tracked_bbox
    counter, fps = 0, 0
    start_time = time.time()

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Visualization parameters
    row_size = 20  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 255)  # red
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    # Initialize the SORT tracker
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

        counter += 1
        # image = cv2.flip(image, 1)

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        input_tensor = vision.TensorImage.create_from_array(rgb_image)
        detection_result = detector.detect(input_tensor)

        dets = []
        categories=[]
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            confidence = detection.categories[0].score
            category=detection.categories[0].category_name
            if category != 'person' or confidence<0.4:
              continue
            dets.append([bbox.origin_x, bbox.origin_y, bbox.origin_x+bbox.width, bbox.origin_y+bbox.height, confidence])
        # Update SORT tracker
        tracked_objects = tracker.update(np.array(dets))

        # Draw tracked objects
        max_id, max_conf = None, 0
        present = False
        best_box=[]
        for i,to in enumerate(tracked_objects):
            bbox = to[:4].astype(int)
            # print(to)
            object_id = int(to[4])
            object_conf=dets[i][4]
            if (object_id == lock_id):
              present = True
              cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
              tracked_bbox=bbox
              cv2.putText(image, f"ID: {object_id} Confidence: {object_conf} ", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
              break

            if (max_conf < object_conf):
              max_conf = object_conf
              max_id = object_id
              best_box=bbox
            
        if (not present or lock_id is None):
          lock_id = max_id
          tracked_bbox=best_box

        if len(tracked_bbox) == 4:
          print(tracked_bbox)
          center=tracked_bbox[0]+(tracked_bbox[2]-tracked_bbox[0])//2
          offset=center-320
          param4=1
          print(f"offset:{offset} center:{center}")
          if abs(offset) < 5:
            param1=0
            param2=0
            param3=0
          else:
            param1,param2,param3 = get_params(offset)
          cv2.putText(image, f"Yaw degree: {param1:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
          cv2.putText(image, f"Yaw speed: {param2:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)




        # Calculate and show FPS
        # if counter % fps_avg_frame_count == 0:
        #     end_time = time.time()
        #     fps = fps_avg_frame_count / (end_time - start_time)
        #     start_time = time.time()

        # fps_text = 'FPS = {:.1f}'.format(fps)
        # text_location = (left_margin, row_size)
        # cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
        #             font_size, text_color, font_thickness)

        # Show the processed image

        cv2.imshow('object_detector', image)

        # Exit on ESC key
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU))


if __name__ == '__main__':
  main()
