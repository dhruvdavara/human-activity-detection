import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from model import CNNLSTM


MODEL_PATH = "best_cnn_lstm.pth"

NUM_FRAMES = 16
IMAGE_SIZE = 224

CLASS_NAMES = [
    "Boxing",
    "Handclapping",
    "Handwaving",
    "Jogging",
    "Running"
]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

eval_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model():
    model = CNNLSTM(
        num_classes=6,
        hidden_size=256,
        num_layers=2,
        dropout=0.3
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


model = load_model()


def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        cap.release()
        raise RuntimeError(
            f"Could not read video: {video_path}"
        )

    indices = np.linspace(
        0,
        total_frames - 1,
        NUM_FRAMES
    ).astype(int)

    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()

        if not ret:
            if frames:
                frame = frames[-1].copy()
            else:
                frame = np.zeros(
                    (120, 160, 3),
                    dtype=np.uint8
                )
        else:
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

        frames.append(frame)

    cap.release()
    return frames


def preprocess_video(video_path):
    frames = extract_frames(video_path)

    processed_frames = [
        eval_transform(frame)
        for frame in frames
    ]

    # [16, 3, 224, 224]
    return torch.stack(processed_frames)


def predict_video(video_path):
    video = preprocess_video(video_path)
    video = video.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(video)

        # The trained checkpoint has 6 output neurons,
        # but this project has only 5 real classes.
        outputs = outputs[:, :5]

        probabilities = F.softmax(outputs, dim=1)

        predicted_index = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0, predicted_index
        ].item()

    return {
        "activity": CLASS_NAMES[predicted_index],
        "confidence": confidence,
        "probabilities": {
            CLASS_NAMES[i]: float(probabilities[0, i])
            for i in range(5)
        }
    }
