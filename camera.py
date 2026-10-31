from functools import lru_cache
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential


BASE_DIR = Path(__file__).resolve().parent

EMOTIONS = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised",
}

SONG_FILES = {
    0: "angry.csv",
    1: "disgusted.csv",
    2: "fearful.csv",
    3: "happy.csv",
    4: "neutral.csv",
    5: "sad.csv",
    6: "surprised.csv",
}

SONG_COLUMNS = ["Name", "Album", "Artist"]

# The current Flask app shares recommendations across its local session.
current_emotion = 4
prediction_lock = Lock()


def build_model():
    """Recreate the architecture used in train.py."""
    model = Sequential(
        [
            Input(shape=(48, 48, 1)),
            Conv2D(32, kernel_size=(3, 3), activation="relu"),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Conv2D(128, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(1024, activation="relu"),
            Dropout(0.5),
            Dense(7, activation="softmax"),
        ]
    )

    weights_path = BASE_DIR / "model.h5"

    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}"
        )

    model.load_weights(str(weights_path))
    return model


emotion_model = build_model()


@lru_cache(maxsize=7)
def load_songs(emotion_index):
    """Read and validate each song list once."""
    path = BASE_DIR / SONG_FILES[emotion_index]

    if not path.is_file():
        raise FileNotFoundError(f"Song file not found: {path}")

    songs = pd.read_csv(path)
    missing_columns = set(SONG_COLUMNS) - set(songs.columns)

    if missing_columns:
        raise ValueError(
            f"{path.name} is missing columns: "
            f"{', '.join(sorted(missing_columns))}"
        )

    return songs.loc[:, SONG_COLUMNS].head(15).copy()


def music_rec(emotion_index=None):
    """Return songs for the selected or most recent expression."""
    if emotion_index is None:
        emotion_index = current_emotion

    if emotion_index not in EMOTIONS:
        raise ValueError(f"Unknown emotion index: {emotion_index}")

    return load_songs(emotion_index).copy()


class VideoCamera:
    """Capture frames and return annotated images with song suggestions."""

    def __init__(self, src=0):
        self.capture = None
        self.emotion_index = 4

        cascade_path = (
            Path(cv2.data.haarcascades)
            / "haarcascade_frontalface_default.xml"
        )
        self.face_cascade = cv2.CascadeClassifier(str(cascade_path))

        if self.face_cascade.empty():
            raise RuntimeError("Could not load the face detector.")

        # Check all song files before opening the camera.
        for emotion_index in EMOTIONS:
            load_songs(emotion_index)

        self.capture = cv2.VideoCapture(src)

        if not self.capture.isOpened():
            self.close()
            raise RuntimeError(
                "Could not open the webcam. Check camera permissions "
                "and close other apps using it."
            )

    def get_frame(self):
        global current_emotion

        if self.capture is None:
            raise RuntimeError("The webcam connection is closed.")

        success, frame = self.capture.read()

        if not success or frame is None:
            raise RuntimeError(
                "Could not read a webcam frame. Check the camera "
                "connection and restart the app."
            )

        # Preserve the camera image's aspect ratio.
        height, width = frame.shape[:2]
        display_width = 600
        display_height = max(1, round(height * display_width / width))
        frame = cv2.resize(frame, (display_width, display_height))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
        )

        if len(faces) > 0:
            # Use the largest detected face for one consistent song list.
            x, y, w, h = max(
                faces,
                key=lambda rectangle: rectangle[2] * rectangle[3],
            )

            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (48, 48))

            # Match the rescale=1./255 preprocessing in train.py.
            model_input = face.astype(np.float32) / 255.0
            model_input = model_input.reshape(1, 48, 48, 1)

            with prediction_lock:
                predictions = emotion_model.predict(
                    model_input,
                    verbose=0,
                )
                self.emotion_index = int(np.argmax(predictions[0]))
                current_emotion = self.emotion_index

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                EMOTIONS[self.emotion_index],
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame,
                "No face detected",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Keep the previous recommendations when no face is visible.
        recommendations = music_rec(self.emotion_index)

        success, jpeg = cv2.imencode(".jpg", frame)

        if not success:
            raise RuntimeError("Could not encode the webcam frame.")

        return jpeg.tobytes(), recommendations

    def close(self):
        """Release the camera when streaming ends."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __del__(self):
        self.close()
