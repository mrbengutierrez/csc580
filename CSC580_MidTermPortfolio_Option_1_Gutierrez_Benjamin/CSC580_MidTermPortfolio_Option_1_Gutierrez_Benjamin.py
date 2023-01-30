import cv2
import dlib
import face_recognition

# Load the individual face image
individual_image = cv2.imread("individual_face.jpg")

# Load the group of faces image
group_image = cv2.imread("group_faces.jpg")

# Use dlib's face detector
detector = dlib.get_frontal_face_detector()

# Detect faces in the individual image
individual_faces = detector(individual_image, 1)

# Detect faces in the group image
group_faces = detector(group_image, 1)

# Encode the individual face
individual_face_encoding = face_recognition.face_encodings(individual_image, individual_faces)

# Encode the faces in the group image
group_face_encodings = face_recognition.face_encodings(group_image, group_faces)

# Compare the encoding of individual face with encoding of faces in the group
match = face_recognition.compare_faces(group_face_encodings, individual_face_encoding)

# Check if there is a match
if True in match:
    print("Individual face is present in the group")
else:
    print("Individual face is not present in the group")
