import cv2
import numpy as np


# Callback for nothing (required for trackbars)
def nothing(x):
    pass


# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Create a window for sliders
cv2.namedWindow("Trackbars")

# Create trackbars for minimum and maximum HSV ranges
cv2.createTrackbar("Min Hue", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("Max Hue", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("Min Sat", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("Max Sat", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Min Val", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("Max Val", "Trackbars", 255, 255, nothing)

# Initialize variables for selection
selection_started = False
start_point = None
end_point = None
selected_roi = None


# Function to select an area (similar to the previous ROI selection)
def select_roi(event, x, y, flags, param):
    global start_point, end_point, selected_roi, selection_started

    if event == cv2.EVENT_LBUTTONDOWN:
        selection_started = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if selection_started:
            end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        selection_started = False
        end_point = (x, y)
        selected_roi = (start_point[0], start_point[1], end_point[0], end_point[1])


# Set the mouse callback for ROI selection
cv2.setMouseCallback("Trackbars", select_roi)

# Start capture loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get current positions of the sliders
    min_h = cv2.getTrackbarPos("Min Hue", "Trackbars")
    max_h = cv2.getTrackbarPos("Max Hue", "Trackbars")
    min_s = cv2.getTrackbarPos("Min Sat", "Trackbars")
    max_s = cv2.getTrackbarPos("Max Sat", "Trackbars")
    min_v = cv2.getTrackbarPos("Min Val", "Trackbars")
    max_v = cv2.getTrackbarPos("Max Val", "Trackbars")

    # Define the HSV range based on slider values
    lower_hsv = np.array([min_h, min_s, min_v])
    upper_hsv = np.array([max_h, max_s, max_v])

    # Create a mask for the defined HSV range
    mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)

    # Show the mask
    cv2.imshow("Mask", mask)

    # Draw the selected ROI if it's being drawn
    if selection_started and start_point and end_point:
        cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)

    # If the ROI is selected and confirmed, save it along with HSV values
    if selected_roi and not selection_started:
        x1, y1, x2, y2 = selected_roi
        roi = frame[y1:y2, x1:x2]

        # Save the ROI as an image file
        cv2.imwrite('saved_roi.png', roi)

        # Save the HSV range to a file
        np.savez('hsv_range.npz', lower_hsv=lower_hsv, upper_hsv=upper_hsv)
        print("ROI and HSV range saved!")

        selected_roi = None

    # Show the live webcam feed
    cv2.imshow("Trackbars", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
cv2.destroyAllWindows()
