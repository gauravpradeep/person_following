from pymavlink import mavutil
import numpy as np
import cv2
import time

# Function to attempt connection
def attempt_connection(port):
    try:
        mav = mavutil.mavlink_connection(port)
        mav.wait_heartbeat()
        print(f"Connected to flight controller on {port}")
        return mav
    except Exception as e:
        print(f"Failed to connect on {port}: {e}")
        return None

# Function to get IMU data
def get_imu_data(mav):
    try:
        msg = mav.recv_match(type='ATTITUDE', blocking=True, timeout=5)
        if msg is not None:
            roll = msg.roll * (180.0 / np.pi)  # Convert radians to degrees
            return roll
    except Exception as e:
        print(f"Error reading IMU data: {e}")
        return None

# Function to rotate image
def rotate_image(image, angle):
    height, width = image.shape[:2]
    image_center = (width / 2, height / 2)
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]
    rotated_mat = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h))
    return rotated_mat

# Function to correct frame for roll only
def correct_frame_for_roll(frame, roll):
    corrected_frame = rotate_image(frame, -roll)  # Negate the angle for correction
    return corrected_frame

# Function to crop the frame to 50% of its original size, centered
def crop_center(frame):
    h, w = frame.shape[:2]
    cropped_frame = frame[h//4:h*3//4, w//4:w*3//4]
    return cropped_frame

# List of serial ports to try
ports = ['/dev/ttyACM0', '/dev/ttyACM1']

# Try to establish a connection
for port in ports:
    mav = attempt_connection(port)
    if mav is not None:
        break

if mav is None:
    print("Failed to connect to any flight controller. Exiting.")
    exit(1)

# Initialize the webcam
cap = cv2.VideoCapture(0)

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Get IMU data (roll only)
        roll = get_imu_data(mav)
        if roll is not None:
            # Correct the frame for roll
            corrected_frame = correct_frame_for_roll(frame, roll)

            # Crop the corrected frame
            cropped_frame = crop_center(corrected_frame)

            # Display the resulting cropped frame
            cv2.imshow('Cropped Stabilized Frame', cropped_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Script interrupted by user")
finally:
    # Release the capture and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
