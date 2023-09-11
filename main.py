import cv2
from comparision import SimpleFacerec

# Initialize the face recognition model
face_recognizer = SimpleFacerec()
face_recognizer.load_encoding_images("images/")

# Open the camera with index 0 (assuming it's your primary camera)
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Camera not found or cannot be opened.")
    exit()

while True:
    ret, frame = camera.read()

    if not ret:
        print("Error: Couldn't read frame from the camera.")
        break

    # Detect faces in the frame
    detected_faces, face_names = face_recognizer.detect_known_faces(frame)
    for face_loc, name in zip(detected_faces, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()
