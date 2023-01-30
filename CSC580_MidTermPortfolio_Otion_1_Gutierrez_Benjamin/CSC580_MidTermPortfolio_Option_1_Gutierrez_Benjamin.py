import cv2
import face_recognition

# Load the individual face image
individual_face = cv2.imread("individual_face.jpg")

# Load the group of faces image
group_faces = cv2.imread("group_faces.jpg")

# Find the face locations in the individual face image
individual_face_locations = face_recognition.face_locations(individual_face)

# Find the face locations in the group of faces image
group_faces_locations = face_recognition.face_locations(group_faces)

# Encode the individual face
individual_face_encoding = face_recognition.face_encodings(individual_face, individual_face_locations)[0]

# Encode the faces in the group image
group_face_encodings = face_recognition.face_encodings(group_faces, group_faces_locations)

# Compare the encoding of individual face with encoding of faces in the group
match = face_recognition.compare_faces(group_face_encodings, individual_face_encoding)

# Check if there is a match
if True in match:
    print("Individual face is present in the group")
else:
    print("Individual face is not present in the group")