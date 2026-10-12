from pathlib import Path
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "face_model.yml"
LABELS_PATH = MODEL_DIR / "labels.txt"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.secret_key = "change-this-development-key"
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def load_labels():
    if not LABELS_PATH.exists():
        return {}
    result = {}
    for line in LABELS_PATH.read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            key, value = line.split("\t", 1)
            result[int(key)] = value
    return result

def model_ready():
    return MODEL_PATH.exists() and LABELS_PATH.exists()

def detect_and_recognize(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("OpenCV could not read this image. Try a JPG or PNG file.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(45, 45)
    )
    labels = load_labels()
    recognizer = None

    if model_ready():
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(str(MODEL_PATH))

    results = []
    for (x, y, w, h) in faces:
        identity = "Unknown"
        distance = None
        # LBPH distance is a distance measure, not a calibrated probability.
        if recognizer is not None:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (160, 160))
            predicted_id, distance = recognizer.predict(face_roi)
            # Lower distance means a closer match. Tune this threshold with held-out data.
            if predicted_id in labels and distance <= 75:
                identity = labels[predicted_id]
        cv2.rectangle(image, (x, y), (x+w, y+h), (40, 210, 90), 2)
        label_text = identity
        if distance is not None:
            label_text += f" | distance {distance:.1f}"
        cv2.putText(image, label_text, (x, max(24, y-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 210, 90), 2, cv2.LINE_AA)
        results.append({
            "identity": identity,
            "distance": round(float(distance), 2) if distance is not None else None
        })

    output_name = "result_" + Path(image_path).name
    output_path = UPLOAD_DIR / output_name
    cv2.imwrite(str(output_path), image)
    return output_name, results

@app.route("/", methods=["GET"])
def index():
    people = sorted(p.name for p in (BASE_DIR / "dataset").iterdir() if p.is_dir()) if (BASE_DIR / "dataset").exists() else []
    return render_template("index.html", model_ready=model_ready(), people=people)

@app.route("/recognize", methods=["POST"])
def recognize():
    file = request.files.get("image")
    if not file or not file.filename:
        flash("Please choose an image first.", "error")
        return redirect(url_for("index"))
    if not allowed_file(file.filename):
        flash("Unsupported file type. Upload JPG, JPEG, PNG, or BMP.", "error")
        return redirect(url_for("index"))

    safe_name = secure_filename(file.filename)
    saved_path = UPLOAD_DIR / safe_name
    file.save(saved_path)
    try:
        output_name, results = detect_and_recognize(saved_path)
    except Exception as exc:
        flash(str(exc), "error")
        return redirect(url_for("index"))

    return render_template(
        "results.html",
        image_url=url_for("static", filename="uploads/" + output_name),
        results=results,
        model_ready=model_ready()
    )

@app.route("/about")
def about():
    return render_template("about.html")

@app.errorhandler(413)
def too_large(_error):
    flash("Image is too large. Maximum upload size is 8 MB.", "error")
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
