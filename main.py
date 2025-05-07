import cv2
import time

def main():
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    prev_frame_time = 0
    new_frame_time = 0

    print("✅ Press 'q' to quit the face detection.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw face rectangles and labels
        for (x, y, w, h) in faces:
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(overlay, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            frame = overlay

        # Calculate FPS
        new_frame_time = time.time()
        fps = int(1 / (new_frame_time - prev_frame_time)) if prev_frame_time != 0 else 0
        prev_frame_time = new_frame_time

        # Display instructions and FPS
        cv2.putText(frame, f"Faces Detected: {len(faces)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        cv2.putText(frame, f"FPS: {fps}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Show the frame
        cv2.imshow('Face Detection - Enhanced UI', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Exiting face detection.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
