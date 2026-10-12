from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "face_model.yml"
LABELS_PATH = MODEL_DIR / "labels.txt"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def main():
    if not DATASET_DIR.exists():
        print("Dataset folder not found.")
        return

    detector = cv2.CascadeClassifier(CASCADE_PATH)
    faces, labels = [], []
    label_names = {}
    next_id = 0

    person_dirs = sorted(p for p in DATASET_DIR.iterdir() if p.is_dir())
    if not person_dirs:
        print("Add one folder per consenting participant inside dataset/.")
        print("Example: dataset/Alice/ and dataset/Bob/")
        return

    for person_dir in person_dirs:
        image_paths = sorted(p for p in person_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not image_paths:
            print(f"Skipping {person_dir.name}: no supported images.")
            continue

        person_id = next_id
        next_id += 1
        label_names[person_id] = person_dir.name

        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Skipping unreadable image: {image_path}")
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detected = detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(45, 45)
            )
            if len(detected) == 0:
                print(f"No face detected in {image_path.name}; skipped.")
                continue
            # Use the largest detected face in each training image.
            x, y, w, h = max(detected, key=lambda box: box[2] * box[3])
            roi = cv2.resize(gray[y:y+h, x:x+w], (160, 160))
            faces.append(roi)
            labels.append(person_id)

    if len(faces) < 2 or len(set(labels)) < 2:
        print("Need usable face images from at least TWO participants to train this demo.")
        print("Aim for 10-20 varied images per person; use consented images.")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels, dtype=np.int32))
    recognizer.write(str(MODEL_PATH))
    LABELS_PATH.write_text(
        "".join(f"{key}\t{name}\n" for key, name in label_names.items()),
        encoding="utf-8"
    )
    print(f"Model saved to: {MODEL_PATH}")
    print(f"People trained: {len(set(labels))}")
    print(f"Training face crops used: {len(faces)}")
    print("Important: evaluate on separate images not used during training.")

if __name__ == "__main__":
    main()
