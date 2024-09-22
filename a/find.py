import cv2
import numpy as np

# Load the saved object template (grayscale)
object_template = cv2.imread('object_template.jpg', 0)

# Initialize the ORB detector
orb = cv2.ORB_create()

# Compute keypoints and descriptors for the saved object
kp1, des1 = orb.detectAndCompute(object_template, None)

# Create BFMatcher object for feature matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Open the first available webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the captured frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and descriptors in the current frame
    kp2, des2 = orb.detectAndCompute(gray_frame, None)

    if des2 is not None:
        # Match descriptors between the saved object and the current frame
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # If there are enough good matches, consider it a match
        good_matches_threshold = 10
        if len(matches) > good_matches_threshold:
            # Draw matches
            result = cv2.drawMatches(object_template, kp1, gray_frame, kp2, matches[:20], None, flags=2)

            # Display the matched result with keypoints
            cv2.imshow('Matches - Object Found', result)
        else:
            cv2.imshow('Webcam Live Stream - Object Not Found', frame)
    else:
        cv2.imshow('Webcam Live Stream - Object Not Found', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
