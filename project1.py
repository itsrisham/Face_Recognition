import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)

# Load known face encodings and names
ratan_tata_image = face_recognition.load_image_file("Photos/tata.jpeg")
ratan_tata_encoding = face_recognition.face_encodings(ratan_tata_image)[0]

srk_image = face_recognition.load_image_file("Photos/srk.jpeg")
srk_encoding = face_recognition.face_encodings(srk_image)[0]

risham_image = face_recognition.load_image_file("Photos/goyal.jpg")
risham_encoding = face_recognition.face_encodings(risham_image)[0]

known_face_encodings = [
    ratan_tata_encoding,
    srk_encoding,
    risham_encoding
]

known_face_names = [
    "Ratan Tata",
    "Shah Rukh Khan",
    "Risham Goyal"
]

students = known_face_names.copy()

face_names = []

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date + '.csv', 'w+', newline='')
lnwrite = csv.writer(f)

while True:
    _, frame = video_capture.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for face_recognition

    # Find all face locations in the frame
    face_locations = face_recognition.face_locations(rgb_frame)

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # Check if the face matches any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            best_match_index = matches.index(True)
            name = known_face_names[best_match_index]

        # Remove the detected student from the list
        if name in students:
            students.remove(name)
            current_time = now.strftime("%H-%M-%S")
            lnwrite.writerow([name, current_time])

        # Print only when a face is recognized
        print("Detected:", name)

    face_names.append(name)



    # Display the frame
    cv2.imshow('Video', frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close window
video_capture.release()
cv2.destroyAllWindows()
f.close()
