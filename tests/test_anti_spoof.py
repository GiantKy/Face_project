import sys
import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# =========================
# IMPORT FACE DETECTOR
# =========================
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.face_detection.detector import FaceDetector

detector = FaceDetector()

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# ANTI-SPOOF MODEL
# =========================
model = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Anti_Spoof.pt")

checkpoint = torch.load(MODEL_PATH, map_location=device)

if isinstance(checkpoint, dict) and "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# PREDICT
# =========================
def predict(face_img):
    img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)

        fake_prob = probs[0][0].item()
        real_prob = probs[0][1].item()

    return fake_prob, real_prob

# =========================
# CAMERA LOOP
# =========================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not ret:
        break

    faces = detector.detect(frame)

    for face in faces:
        x1, y1, x2, y2 = face["bbox"]

        # safety crop
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        face_img = frame[y1:y2, x1:x2]

        if face_img.size == 0:
            continue

        fake_prob, real_prob = predict(face_img)

        # =========================
        # DECISION
        # =========================
        if real_prob > fake_prob:
            label = "REAL"
            color = (0, 255, 0)
            score = real_prob
        else:
            label = "FAKE"
            color = (0, 0, 255)
            score = fake_prob

        # draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            frame,
            f"{label} {score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow("E-KYC Anti-Spoof Pipeline", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()