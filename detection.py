# dataloader.py

from pathlib import Path
import face_recognition
import pickle
from tqdm import tqdm
from collections import Counter
from PIL import Image, ImageDraw
import argparse


DEFAULT_ENCODING_PATH = Path("output/encodings.pkl")

Path("output").mkdir(exist_ok=True)
Path("training").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)


def parse_the_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--if_train",
        default=False,
        help="If you want to train, you need to set this flag to True",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="hog",
        help="Choose the model to detect the face",
    )
    parser.add_argument(
        "-e",
        "--encoding_path",
        default=DEFAULT_ENCODING_PATH,
        help="The path to save the encoding file",
    )
    parser.add_argument(
        "-t",
        "--training_path",
        default=Path("training"),
        help="The path of the training data",
    )
    parser.add_argument(
        "-u",
        "--unknown",
        default="validation/geral/微信图片_20240419150729.jpg",
        help="The path of the unknown face image, which you want to recognize",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output/recognized.jpg",
        help="The path of the output image",
    )
    return parser.parse_args()


def encode_known_face(
    model: str = "hog",
    encoding_path: Path = DEFAULT_ENCODING_PATH,
    training_path: Path = Path("training"),
) -> None:
    known_face_encodings = []
    known_face_names = []

    # loop the every known-face user directory
    for image_path in training_path.glob("*/*"):
        name = image_path.parent.name
        image = face_recognition.load_image_file(image_path)

        face_locations = face_recognition.face_locations(
            image, model=model)  # get the face location in the image
        face_encodings = face_recognition.face_encodings(
            image, face_locations)  # get the face encoding

        for encoding in face_encodings:
            known_face_encodings.append(encoding)
            known_face_names.append(name)
    names_encodings = {"names": known_face_names,
                       "encodings": known_face_encodings}
    with open(encoding_path, "wb") as f:
        pickle.dump(names_encodings, f)


def compare_faces(
    unknown_face_encoding,
    names_encodings,
    threshold: float = 0.0,
) -> str:
    # compare the unknown face encoding with known face encoding
    matches = face_recognition.compare_faces(
        names_encodings["encodings"], unknown_face_encoding)

    # vote the name of the known face
    votes = Counter(
        [name
         for name, match in zip(
             names_encodings["names"], matches)
         if match]
    )
    if votes:
        name = votes.most_common(1)[0][0]
        times = votes.most_common(1)[0][1]
        if times / len(matches) >= threshold:
            return name
        else:
            return "Unknown"

    return "Unknown"


def display_face(
    draw: ImageDraw,
    bounding_box: tuple,
    name: str,
) -> None:
    top, right, bottom, left = bounding_box
    # draw bounding box
    draw.rectangle(
        [(left, top), (right, bottom)],
        outline="red",
        width=3,
    )
    # draw name
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom), name)
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill="blue",
        outline="blue",
    )
    draw.text(
        (left, bottom),
        name,
        fill="white",
    )


def recognize_unknown_face(
    image_path: Path,
    model: str = "hog",
    encoding_path: Path = DEFAULT_ENCODING_PATH,
) -> None:
    # read the encoding file(trained data)
    with open(encoding_path, "rb") as f:
        names_encodings = pickle.load(f)

    unknown_image = face_recognition.load_image_file(image_path)
    unknown_face_locations = face_recognition.face_locations(
        unknown_image, model=model)
    unknown_face_encodings = face_recognition.face_encodings(
        unknown_image, unknown_face_locations)

    # transfer into a pillow image
    pillow_image = Image.fromarray(unknown_image)
    draw = ImageDraw.Draw(pillow_image)

    locations_encodings = zip(unknown_face_locations, unknown_face_encodings)
    for bounding_box, unknown_encoding in locations_encodings:
        name = compare_faces(unknown_encoding, names_encodings)

        if name == "unknown":

            continue
        else:
            display_face(draw, bounding_box, name)

    del draw
    pillow_image.save("output/recognized.jpg")


if __name__ == "__main__":
    # parse the args
    args = parse_the_args()
    model = args.model
    if_train = args.if_train
    encoding_path = args.encoding_path
    training_path = args.training_path
    unknown_image_path = args.unknown

    # train the model if the flag is true
    if if_train == "True":
        encode_known_face(
            model=model,
            encoding_path=encoding_path,
            training_path=training_path,
        )
        print("Training Completed")

    # recognize unknown face
    recognize_unknown_face(
        unknown_image_path,
        model=model,
        encoding_path=encoding_path,
    )
