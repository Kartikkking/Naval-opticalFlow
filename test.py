import cv2
import numpy as np

class ObjectTracker:
    def __init__(self):
        # Initialize object detection
        self.object_detector = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50)
        
        # Optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Initialize tracking points
        self.prev_frame = None
        self.prev_points = None

    def detect_objects(self, frame):
        # Apply background subtraction
        mask = self.object_detector.apply(frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 10:  # Lower threshold to detect smaller objects
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append([x, y, w, h])
        
        return detections

    def track_optical_flow(self, frame_gray):
        if self.prev_frame is None:
            self.prev_frame = frame_gray
            return None
        
        if self.prev_points is None:
            # Detect initial features to track
            self.prev_points = cv2.goodFeaturesToTrack(frame_gray, 
                                                      maxCorners=100,
                                                      qualityLevel=0.2,
                                                      minDistance=5,
                                                      blockSize=5)
        
        if self.prev_points is not None:
            # Calculate optical flow
            new_points, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_frame, frame_gray, self.prev_points, None, **self.lk_params)
            
            if new_points is not None:
                good_new = new_points[status == 1]
                good_old = self.prev_points[status == 1]
                
                self.prev_points = good_new.reshape(-1, 1, 2)
                self.prev_frame = frame_gray
                
                return good_new, good_old
        
        self.prev_frame = frame_gray
        return None

def process_camera_stream():
    tracker = ObjectTracker()
    
    # Ask user for input method
    input_method = input("Enter '1' to use webcam or '2' to upload a video: ")
    
    if input_method == '1':
        cap = cv2.VideoCapture(0)  # Use default camera
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
    elif input_method == '2':
        video_path = input("Enter the path to the video file: ")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return
    else:
        print("Invalid input method. Exiting.")
        return
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect objects
            detections = tracker.detect_objects(frame)
            
            # Draw detection boxes
            for box in detections:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, 'Object', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Track optical flow
            flow_result = tracker.track_optical_flow(frame_gray)
            if flow_result is not None:
                good_new, good_old = flow_result
                
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    frame = cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 2)
                    frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)
            
            # Show the real-time frame
            cv2.imshow('Object Tracker', frame)
            
            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

# Start real-time camera object tracking
process_camera_stream()
