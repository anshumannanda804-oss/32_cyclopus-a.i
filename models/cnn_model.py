"""
CycloneImageClassifier — PyTorch CNN for Satellite IR Imagery Classification.

Architecture:
  3 × (Conv2d → BatchNorm2d → ReLU → MaxPool2d) → AdaptiveAvgPool → FC(512→128→5)

Classes (IMD Scale):
  0: No Cyclone
  1: Depression
  2: Cyclonic Storm
  3: Severe Cyclonic Storm
  4: Very Severe Cyclonic Storm+

Input: 128×128 single-channel (grayscale IR) images
"""

import os
import io
import pickle
import numpy as np
from typing import Dict, Any, Optional, List

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ──────────────────────────────────────────────────────────
# IMD Intensity Class Labels
# ──────────────────────────────────────────────────────────
CLASS_LABELS: List[str] = [
    "No Cyclone",
    "Depression",
    "Cyclonic Storm",
    "Severe Cyclonic Storm",
    "Very Severe Cyclonic Storm+"
]

NUM_CLASSES = len(CLASS_LABELS)
INPUT_SIZE = 128  # 128×128 pixels


# ──────────────────────────────────────────────────────────
# CNN Architecture
# ──────────────────────────────────────────────────────────
if TORCH_AVAILABLE:
    class _CycloneCNN(nn.Module):
        """3-block CNN for cyclone IR image classification."""

        def __init__(self, num_classes: int = NUM_CLASSES):
            super().__init__()

            # Block 1: 1 → 32 channels
            self.block1 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),          # 128 → 64
                nn.Dropout2d(0.25)
            )

            # Block 2: 32 → 64 channels
            self.block2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),          # 64 → 32
                nn.Dropout2d(0.25)
            )

            # Block 3: 64 → 128 channels
            self.block3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(4),  # → 4×4
                nn.Dropout2d(0.25)
            )

            # Classifier head
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.classifier(x)
            return x


# ──────────────────────────────────────────────────────────
# High-Level Wrapper (used by backend API)
# ──────────────────────────────────────────────────────────
class CycloneImageClassifier:
    """
    User-facing wrapper that loads weights and provides
    predict_from_image / predict_from_array convenience methods.
    """

    def __init__(self, weights_path: Optional[str] = None):
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not installed. Run: pip install torch torchvision"
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _CycloneCNN(num_classes=NUM_CLASSES).to(self.device)
        self.model.eval()
        self.class_labels = CLASS_LABELS

        # Standard transform pipeline for inference
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),           # [0, 255] → [0.0, 1.0]
            transforms.Normalize([0.5], [0.5])  # → [−1, 1]
        ])

        if weights_path and os.path.exists(weights_path):
            self.load_weights(weights_path)

    # ── Weight Management ──────────────────────────────────
    def load_weights(self, path: str):
        """Load .pth state dict."""
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    def save_weights(self, path: str):
        """Save .pth state dict."""
        torch.save(self.model.state_dict(), path)

    def save_as_pkl(self, path: str):
        """Save full wrapper as pickle (for quick deploy / testing)."""
        with open(path, "wb") as f:
            pickle.dump({
                "class_labels": self.class_labels,
                "state_dict": self.model.state_dict()
            }, f)

    @classmethod
    def load_from_pkl(cls, path: str) -> "CycloneImageClassifier":
        """Load a pickle-exported model."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.model.load_state_dict(data["state_dict"])
        obj.class_labels = data["class_labels"]
        obj.model.eval()
        return obj

    # ── Inference ──────────────────────────────────────────
    def predict_from_image(self, image_path: str) -> Dict[str, Any]:
        """Classify a satellite image file (PNG/JPG/TIFF)."""
        img = Image.open(image_path).convert("L")  # to grayscale
        return self._run_inference(img)

    def predict_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """Classify from raw bytes (e.g. base64 decoded upload)."""
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        return self._run_inference(img)

    def predict_from_array(self, arr: np.ndarray) -> Dict[str, Any]:
        """
        Classify from a NumPy array.
        Expected shape: (H, W) uint8 grayscale or (H, W, 1).
        """
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        img = Image.fromarray(arr.astype(np.uint8), mode="L")
        return self._run_inference(img)

    def _run_inference(self, pil_image: "Image.Image") -> Dict[str, Any]:
        """Core inference pipeline."""
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)  # (1, 1, 128, 128)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)  # (NUM_CLASSES,)

        probs_np = probs.cpu().numpy()
        predicted_idx = int(np.argmax(probs_np))

        return {
            "predicted_class": self.class_labels[predicted_idx],
            "predicted_index": predicted_idx,
            "confidence": round(float(probs_np[predicted_idx]) * 100, 2),
            "class_probabilities": {
                self.class_labels[i]: round(float(probs_np[i]) * 100, 2)
                for i in range(NUM_CLASSES)
            }
        }

    # ── Training Helper (returns raw model for train_cnn.py) ──
    def get_raw_model(self) -> "nn.Module":
        return self.model
