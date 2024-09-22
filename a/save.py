import cv2

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

    # Display the frame
    cv2.imshow('Webcam Stream - Press "s" to save the object', frame)

    # Wait for user input to save the object
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # Save the current frame as the object template
        cv2.imwrite('object_template.jpg', frame)
        print("Object saved as 'object_template.jpg'")
        break
    elif key == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
