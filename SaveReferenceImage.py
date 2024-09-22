import cv2
import numpy as np

# Open webcam
cap = cv2.VideoCapture(0)

# Function to capture an object (e.g., a creeper)
def capture_template():
    ret, frame = cap.read()
    if ret:
        # Select ROI (Region of Interest)
        roi = cv2.selectROI("Select Object to Scan", frame, showCrosshair=True)
        if roi != (0, 0, 0, 0):
            # Crop the region of interest and save it as the template
            template = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
            cv2.imwrite('creeper_template.jpg', template)
            print("Template captured and saved as 'creeper_template.jpg'")
    cap.release()
    cv2.destroyAllWindows()

capture_template()
