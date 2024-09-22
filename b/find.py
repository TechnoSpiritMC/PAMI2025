import cv2
import numpy as np

# Load the saved HSV range and the object image
data = np.load('hsv_range.npz')
lower_hsv = data['lower_hsv']
upper_hsv = data['upper_hsv']
template = cv2.imread('saved_roi.png', 0)  # Load as grayscale for shape matching

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask based on the saved HSV range
    mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)

    # Apply the mask to get the detected area
    detected_area = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert detected area to grayscale for shape matching
    gray_frame = cv2.cvtColor(detected_area, cv2.COLOR_BGR2GRAY)

    # Use template matching to detect the object (shape comparison)
    res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the match is good enough, draw a rectangle around the detected area
    threshold = 0.5  # Adjust threshold for better detection
    if max_val >= threshold:
        h, w = template.shape
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(frame, 'Detected!', (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the result
    cv2.imshow('Object Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
cv2.destroyAllWindows()
