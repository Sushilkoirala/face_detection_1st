import face_recognition
import cv2
import os
import glob
import numpy as np

class SimpleFacerec:
    def __init__(self):
        # Initialize lists to store known face encodings and corresponding names
        self.known_face_encodings = []
        self.known_face_names = []

        # Resize frame for faster processing
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_dir):
        """
        Load face encoding images from a directory.
        :param images_dir: The directory containing face encoding images.
        """
        image_files = glob.glob(os.path.join(images_dir, "*.*"))

        print(f"{len(image_files)} face encoding images found.")

        for image_file in image_files:
            image = cv2.imread(image_file)
            if image is None:
                print(f"Skipping {image_file} as it could not be loaded.")
                continue

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            base_name = os.path.basename(image_file)
            name, _ = os.path.splitext(base_name)

            # Get the face encoding
            face_encodings = face_recognition.face_encodings(rgb_image)

            if not face_encodings:
                print(f"No face found in {image_file}. Skipping.")
                continue

            # Store the file name and face encoding
            self.known_face_encodings.append(face_encodings[0])
            self.known_face_names.append(name)

        print("Face encoding images loaded")

    def detect_known_faces(self, frame):
        # Resize the input frame
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find face locations and encodings in the small frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        recognized_names = []
        for face_encoding in face_encodings:
            # Compare the face encoding to known face encodings
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # Find the best match among known faces
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            recognized_names.append(name)

        # Convert face locations to adjusted coordinates based on frame resizing
        face_locations = np.array(face_locations)
        face_locations = (face_locations / self.frame_resizing).astype(int)

        return face_locations, recognized_names
