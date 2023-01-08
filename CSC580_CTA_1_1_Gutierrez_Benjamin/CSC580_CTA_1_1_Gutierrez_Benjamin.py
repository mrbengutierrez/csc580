"""
Student: Benjamin Gutierrez
Date: December 18, 2022
Course: Applying Machine Learning and Neural Networks - Capstone
Instructor: Lori Farr
"""

from PIL import Image, ImageDraw
import face_recognition

if __name__ == "__main__":
    image = Image.open("original.jpg")
    image_array = face_recognition.load_image_file("original.jpg")
    face_locations = face_recognition.face_locations(image_array)
    number_of_faces = len(face_locations)
    print(f"Found {number_of_faces} face(s) in this picture.")

    for (top, right, bottom, left) in face_locations:
        print(f"A face is located at pixel location Top: {top}, Left {left},Bottom: {bottom}, Right: {right}")
        draw = ImageDraw.Draw(image)
        draw.rectangle([(left, top), (right, bottom)], outline="green", width=8)
    image.show()
    image.save("final.jpg")
