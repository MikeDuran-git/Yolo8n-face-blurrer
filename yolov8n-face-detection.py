from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2


# Load the model
print("[INFO] loading model...")
model = YOLO("yolov8n-face.pt")
names = model.names # Get class name
#
print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Blur ratio
blur_ratio = 50

print("[INFO] processing video stream...")
try:
    # Read video frame by frame
    while cap.isOpened():
        #success is a boolean that returns True if the frame is read correctly, and False otherwise
        #frame is the frame that was read
        success, frame = cap.read()
        # If the frame is empty, the video processing has been successfully completed
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        # Perform object detection
        #we use the predict method to perform object detection on the frame
        #show=False is used to prevent the frame from being displayed
        results = model.predict(frame, show=False)
        #we get the boxes and classes from the results
        #the boxes are the zones where the objects are detected
        boxes = results[0].boxes.xyxy.cpu().tolist()
        #this is the class of the detected object in the boxes (it can be a person, a car, a dog, etc.)
        #in our case, it is a face (class 0) or a mask (class 1).
        clss = results[0].boxes.cls.cpu().tolist()
        #we create an annotator object to display the boxes and classes on the frame
        annotator = Annotator(frame, line_width=2, example=names)
        #we loop through the boxes and classes to display them on the frame
        if boxes is not None:
            for box, cls in zip(boxes, clss):
                #we use the box_label method to display the boxes and classes on the frame
                annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])
                #we blur the detected object
                obj = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                blur_obj = cv2.blur(obj, (blur_ratio, blur_ratio))
                #we replace the detected object with the blurred object
                frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = blur_obj
        #we display the frame with the boxes and classes
        cv2.imshow("Camera 0", frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt: # Ctrl+C
    print("Program interrupted by user, exiting gracefully...")
except Exception as e: # Any other exception
    print(e)
finally: # Release the VideoCapture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
