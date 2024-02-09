import argparse
import sys
import time
import cv2
import numpy as np
from sort import Sort
from pymavlink import mavutil
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
import torch
from torchvision import transforms
from PIL import Image

def set_mode(master, mode):
    """
    Sets the flight mode
    """
    # mode_id is the numerical representation of the flight mode
    mode_id = master.mode_mapping()[mode]
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id)

    # Wait for mode to change
    while True:
        # Check if mode changed
        if master.flightmode == mode:
            print(f"Mode changed to {mode}")
            break
        time.sleep(1)

def arm_and_takeoff(master, aTargetAltitude):
    """
    Arms vehicle and fly to aTargetAltitude.
    """
    print("Basic pre-arm checks")
    # Don't try to arm until autopilot is ready
    # while not master.motors_armed():
    #     print(" Waiting for vehicle to initialise...")
    #     time.sleep(1)

    print("Arming motors")
    # Copter should arm in GUIDED mode
    

    # Confirm vehicle armed before attempting to take off
    # while not master.motors_armed():
    master.mav.command_long_send(
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0, 1, 0, 0, 0, 0, 0, 0)
    print(" Waiting for arming...")
    time.sleep(0.5)

    print("Taking off!")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, aTargetAltitude)

    # Wait until the vehicle reaches a safe height
    while True:
        print(" Altitude: ", master.location().alt)
        if master.location().alt >= aTargetAltitude * 0.90:  # Trigger just below target alt.
            print("Reached target altitude")
            break
        time.sleep(1)

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
    P_yaw_speed = 0.015

    param1 = P_yaw_deg * offset
    param2 = P_yaw_speed * offset
    param3 = offset / abs(offset)

    return param1, param2, param3


def run():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("Using device:", device)
	lock_id = None
	tracked_bbox = []
	weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
	model = ssdlite320_mobilenet_v3_large(weights=weights).to(device)
	model.eval()
	preprocess = weights.transforms()
	cap = cv2.VideoCapture(0)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
	row_size = 20  # pixels
	left_margin = 24  # pixels
	text_color = (0, 0, 255)  # red
	font_size = 1
	font_thickness = 1
	fps_avg_frame_count = 10
	tracker=Sort()
	counter, fps = 0,0
	start_time = time.time()
	while cap.isOpened():
	    success, image = cap.read()
	    if not success:
	        sys.exit('ERROR: Unable to read from webcam. Please verify your webcam settings.')
	    counter+=1

	    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	    pil_image = Image.fromarray(rgb_image)
	    batch = preprocess(pil_image).unsqueeze(0).to(device)
	    predictions=model(batch)[0]
	    dets=[]


	    for i in range(len(predictions['boxes'])):
	    	boxes = predictions['boxes'][i].detach().cpu().numpy()  # Detach and convert to numpy
	    	scores = predictions['scores'][i].item()  # Get python number from tensor
	    	labels = predictions['labels'][i].item()
	    	if scores >= 0.3 and labels == 1: 
	    		xmin, ymin, xmax, ymax = boxes
	    		dets.append([xmin, ymin, xmax, ymax, scores])

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
	            cv2.putText(image, f"ID: {object_id} Confidence: {object_conf} ", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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
	            condition_yaw(master, param1, param2, relative=True)  

	        cv2.putText(image, f"Yaw degree: {param1:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
	        cv2.putText(image, f"Yaw speed: {param2:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

	    if counter % fps_avg_frame_count == 0:
	    	end_time = time.time()
	    	fps = fps_avg_frame_count / (end_time - start_time)
	    	start_time = time.time()

	    fps_text = 'FPS = {:.1f}'.format(fps)
	    text_location = (left_margin, row_size)
	    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    font_size, text_color, font_thickness)
	    cv2.imshow('object_detector', image)
	    if cv2.waitKey(1) == 27:
	    	break

	cap.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	run()