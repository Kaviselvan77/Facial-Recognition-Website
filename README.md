# Facial Recognition Website (Python, OpenCV, Flask, LBPH)

A beginner-friendly educational demo that detects faces in uploaded images and compares them with identities enrolled in a locally trained LBPH model.

## Features
- Flask website with image upload
- OpenCV Haar Cascade face detection
- LBPH face recognition
- Unknown label when the model distance exceeds a demo threshold
- Separate training and hold-out evaluation scripts
- Annotated output image

## Important limitations and responsible use
This is a learning prototype, not a secure identity verification system. Haar Cascade and LBPH can fail under changes in lighting, pose, image quality, and camera conditions. The displayed LBPH distance is **not a probability or confidence percentage**. The default threshold (75) is a starting point only and must be tuned using validation data. Do not use this prototype for access control, surveillance, employment, financial decisions, or other high-impact decisions. Only use images with the participants' informed consent and do not publish private face images or trained face models.

## Project structure
```text
Facial_Recognition_Website/
├── app.py
├── train_model.py
├── evaluate_model.py
├── requirements.txt
├── dataset/                 # training images, one folder per person
├── dataset_test/             # separate test images, same folder naming
├── models/                   # generated model and label mapping
├── static/css/style.css
├── static/uploads/            # local uploaded/result images
└── templates/
```

## 1. Setup (Windows / VS Code)
Open this folder in VS Code. Open Terminal > New Terminal and run:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt terminal and run:
```bat
.venv\Scripts\activate.bat
```

Install only one OpenCV package in the environment. `opencv-contrib-python` includes the `cv2.face` LBPH module.

## 2. Prepare a dataset
Create folders like:
```text
dataset/
├── Person_1/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── ...
└── Person_2/
    ├── photo1.jpg
    └── ...
```
Use images of at least two consenting participants. Aim for 10–20 images per person with reasonable variation in lighting and angle. Make sure the face is reasonably clear. Do not use the same or near-duplicate images in training and testing.

## 3. Train the model
```powershell
python train_model.py
```
The script detects the largest face in each image and saves `models/face_model.yml` and `models/labels.txt`. It needs usable faces from at least two people.

## 4. Run the website
```powershell
python app.py
```
Open the local address printed in the terminal, usually `http://127.0.0.1:5000`. Upload a JPG, JPEG, PNG, or BMP image. The result shows detected faces and predicted identity (or Unknown).

If you have not trained a model yet, face detection still works, but identities will be shown as Unknown. Stop the server with Ctrl+C.

## 5. Evaluate with separate images
Create `dataset_test/` with one folder per enrolled person and images that were **not** used for training:
```text
dataset_test/
├── Person_1/
│   └── heldout1.jpg
└── Person_2/
    └── heldout1.jpg
```
Then run:
```powershell
python evaluate_model.py
```
The script reports image-level accuracy, precision/recall/F1, and a confusion matrix. It skips images where no face is detected. For realistic false-accept and false-reject evaluation, add a separately managed set of non-enrolled participants and carefully tune the distance threshold on validation data. Do not report fabricated performance numbers.

## Troubleshooting
- **`module 'cv2' has no attribute 'face'`**: activate this project's virtual environment and run `pip uninstall opencv-python opencv-contrib-python -y`, then `pip install opencv-contrib-python`. Do not install both OpenCV distributions together.
- **No faces detected**: use a clearer, front-facing image with good lighting.
- **All results are Unknown**: train on suitable images, verify the test person is enrolled, and tune the LBPH threshold using validation data.
- **Upload does not work**: use a supported image type and keep it under 8 MB.
- **Port already in use**: stop the other Flask process or change the port in `app.py`.

## Evaluation notes
LBPH `predict()` returns a distance, where smaller is generally a closer match. This is not a calibrated confidence score. Accuracy depends on the dataset and test setup; actual measured results must be reported from your run.

## Suggested submission checklist
- [ ] Train the model successfully
- [ ] Run the website and capture a screenshot
- [ ] Run evaluation on held-out images
- [ ] Update this README with your actual dataset description and measured results
- [ ] Remove private or unconsented images before sharing the repository
- [ ] Add screenshots and code to a public GitHub repository, or submit the ZIP
