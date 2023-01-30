import face_recognition


def is_individual_in_photo(individual_filepath: str, group_filepath: str) -> bool:
    """ Tells whether an individual is in a group photo or not.
        Parameters:
            individual_filepath: filepath of the individual to check
            group_filepath: filepath of a group photo the individual may be in
        Returns:
            True if the indvidual is in the photo, otherwise false
    """

    known_image = face_recognition.load_image_file("individual_face_in_photo.jpg")
    unknown_image = face_recognition.load_image_file("group_faces.jpg")

    known_encoding = face_recognition.face_encodings(known_image)[0]
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

    results = face_recognition.compare_faces([known_encoding], unknown_encoding)

    print(results)

    if True in results:
        return True
    return False


def in_photo():
    in_photo_filepath = "individual_face_in_photo.jpg"
    group_filepath = "group_faces.jpg"

    in_photo_result = is_individual_in_photo(in_photo_filepath, group_filepath)
    print(f"Is {in_photo_filepath} contained in {group_filepath}?")
    print(in_photo_result)


def not_in_photo():
    not_in_photo_filepath = "individual_face_not_in_photo.jpg"
    group_filepath = "group_faces.jpg"

    not_in_photo_result = is_individual_in_photo(not_in_photo_filepath, group_filepath)
    print(f"Is {not_in_photo_filepath} contained in {group_filepath}?")
    print(not_in_photo_result)


if __name__ == "__main__":
    in_photo()
    not_in_photo()
