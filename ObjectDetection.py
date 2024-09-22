import cv2
import numpy as np

# Open the first available webcam
cap = cv2.VideoCapture(0)

# Load the Creeper template (grayscale)
creeper_template = cv2.imread('creeper_template.jpg', 0)

# Initiate ORB detector
orb = cv2.ORB_create()

# Compute keypoints and descriptors for the Creeper template
kp1, des1 = orb.detectAndCompute(creeper_template, None)

# Create BFMatcher object for feature matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range for detecting green pixels (in HSV space)
    lower_green = np.array([35, 75, 75])   # Adjusted lower bound for better detection
    upper_green = np.array([85, 255, 255]) # Upper bound of green color

    # Create a mask for green areas in the frame
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Optional: Blur the mask to remove noise
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Use morphological transformations to enhance the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=2)

    # Find contours (outlines) of the regions containing green pixels
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Variables to store the largest contour and its area
    largest_contour = None
    max_area = 0

    # Find the contour with the largest area
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour

    # If a largest contour is found, draw a filled transparent rectangle and a dark gray border
    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Create a dark gray rectangle border
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 2)  # Dark gray border

        # Create a semi-transparent light gray fill inside the rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (200, 200, 200), -1)  # Light gray fill
        alpha = 0.35  # Transparency factor
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Extract the detected region (in grayscale) for ORB comparison
        detected_region = frame[y:y+h, x:x+w]
        gray_detected_region = cv2.cvtColor(detected_region, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and descriptors in the detected region
        kp2, des2 = orb.detectAndCompute(gray_detected_region, None)

        # Match descriptors between the Creeper template and the detected region
        if des2 is not None:
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            # If a sufficient number of good matches are found, consider it a match
            good_matches_threshold = 10
            if len(matches) > good_matches_threshold:
                text = "Green detected - Object: Creeper"
            else:
                text = "Green detected - Object: Unknown"
        else:
            text = "Green detected - Object: Unknown"

        # Display the text
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show the frame with the detection
    cv2.imshow('Webcam Live Stream - Largest Green Detection', frame)

    # Wait for 'q' key to be pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
