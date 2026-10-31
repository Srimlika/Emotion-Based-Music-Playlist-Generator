from threading import Lock

from flask import Flask, Response, jsonify, render_template

from camera import VideoCamera, music_rec


app = Flask(__name__)

recommendations = music_rec()
recommendations_lock = Lock()

# This local app supports one webcam stream at a time.
camera_lock = Lock()


@app.route("/")
def index():
    return render_template("index.html")


def gen(camera):
    global recommendations

    try:
        while True:
            frame, songs = camera.get_frame()

            with recommendations_lock:
                recommendations = songs.copy()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )
    except Exception:
        app.logger.exception("The webcam stream stopped unexpectedly.")
    finally:
        camera.close()
        camera_lock.release()


@app.route("/video_feed")
def video_feed():
    if not camera_lock.acquire(blocking=False):
        return (
            "The webcam is already in use. Close the other stream "
            "and reload this page.",
            409,
        )

    try:
        camera = VideoCamera()
    except Exception:
        camera_lock.release()
        app.logger.exception("Could not start the webcam.")
        return (
            "Could not start the webcam. Check camera permissions "
            "and the app terminal.",
            503,
        )

    return Response(
        gen(camera),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.route("/t")
def gen_table():
    with recommendations_lock:
        songs = recommendations.copy()

    # Replace missing CSV values with empty strings for valid JSON.
    records = songs.fillna("").to_dict(orient="records")

    response = jsonify(records)
    response.headers["Cache-Control"] = "no-store"
    return response


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=True,
    )
