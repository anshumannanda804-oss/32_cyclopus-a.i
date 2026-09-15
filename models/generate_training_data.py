"""
Generate Synthetic IR Satellite Training Images for Cyclone CNN.

Creates 500 images (100 per class) in ImageFolder layout:
    training_data/
      ├── 0_no_cyclone/
      ├── 1_depression/
      ├── 2_cyclonic_storm/
      ├── 3_severe_cyclonic_storm/
      └── 4_very_severe_plus/

Each image is 128×128 grayscale simulating thermal IR cloud-top patterns:
  - No Cyclone:        Random cloud scatter, no spiral
  - Depression:        Faint circular cloud cluster, no eye
  - Cyclonic Storm:    Defined spiral bands, no clear eye
  - Severe Cyclonic:   Tight spiral, emerging eye
  - Very Severe+:      Compact CDO, clear eye, symmetric spiral arms
"""

import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ──────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────
IMG_SIZE = 128
IMAGES_PER_CLASS = 100
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "training_data")

CLASS_DIRS = [
    "0_no_cyclone",
    "1_depression",
    "2_cyclonic_storm",
    "3_severe_cyclonic_storm",
    "4_very_severe_plus"
]


# ──────────────────────────────────────────────────────────
# Base utilities
# ──────────────────────────────────────────────────────────
def _make_base(bg_val: int = 10, noise_std: float = 8.0) -> np.ndarray:
    """Dark background with slight noise (simulating ocean thermal floor)."""
    img = np.full((IMG_SIZE, IMG_SIZE), bg_val, dtype=np.float32)
    img += np.random.normal(0, noise_std, (IMG_SIZE, IMG_SIZE))
    return np.clip(img, 0, 255)


def _add_radial_gradient(img: np.ndarray, cx: int, cy: int,
                         radius: float, intensity: float) -> np.ndarray:
    """Add a radial bright gradient (simulating cloud mass)."""
    yy, xx = np.ogrid[:IMG_SIZE, :IMG_SIZE]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = np.clip(1.0 - dist / radius, 0, 1) ** 1.5
    img = img + mask * intensity
    return np.clip(img, 0, 255)


def _add_spiral_arms(img: np.ndarray, cx: int, cy: int,
                     num_arms: int, tightness: float,
                     arm_width: float, brightness: float,
                     max_radius: float) -> np.ndarray:
    """Draw logarithmic spiral bands from center."""
    yy, xx = np.meshgrid(np.arange(IMG_SIZE), np.arange(IMG_SIZE), indexing="ij")
    dx = xx - cx
    dy = yy - cy
    r = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dy, dx)

    for arm in range(num_arms):
        arm_offset = 2.0 * math.pi * arm / num_arms
        # Spiral equation: r = a * e^(b*theta)
        spiral_theta = theta - arm_offset
        expected_r = tightness * np.exp(0.15 * spiral_theta)
        diff = np.abs(r - expected_r)
        arm_mask = np.exp(-diff ** 2 / (2 * arm_width ** 2))
        arm_mask *= (r < max_radius).astype(float)
        img = img + arm_mask * brightness

    return np.clip(img, 0, 255)


def _add_eye(img: np.ndarray, cx: int, cy: int,
             eye_radius: float, darkness: float = 20) -> np.ndarray:
    """Punch a dark eye hole in the cloud mass."""
    yy, xx = np.ogrid[:IMG_SIZE, :IMG_SIZE]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    eye_mask = (dist < eye_radius).astype(float)
    # Smooth edge
    edge = np.clip(1.0 - (dist - eye_radius + 2) / 4.0, 0, 1)
    eye_mask = np.maximum(eye_mask, edge * 0.5)
    img = img * (1 - eye_mask) + darkness * eye_mask
    return np.clip(img, 0, 255)


def _pil_blur(arr: np.ndarray, radius: float = 1.5) -> np.ndarray:
    """Apply Gaussian blur via PIL for smooth textures."""
    pil = Image.fromarray(arr.astype(np.uint8), mode="L")
    pil = pil.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(pil, dtype=np.float32)


def _random_center() -> tuple:
    """Slight random offset from true center for realism."""
    cx = IMG_SIZE // 2 + np.random.randint(-8, 9)
    cy = IMG_SIZE // 2 + np.random.randint(-8, 9)
    return cx, cy


# ──────────────────────────────────────────────────────────
# Class-specific generators
# ──────────────────────────────────────────────────────────
def generate_no_cyclone() -> np.ndarray:
    """Random scattered clouds, no organized pattern."""
    img = _make_base(bg_val=12, noise_std=12)
    # Random cloud blobs
    for _ in range(np.random.randint(5, 15)):
        bx = np.random.randint(10, IMG_SIZE - 10)
        by = np.random.randint(10, IMG_SIZE - 10)
        br = np.random.uniform(8, 25)
        bi = np.random.uniform(40, 120)
        img = _add_radial_gradient(img, bx, by, br, bi)
    return _pil_blur(img, radius=2.0)


def generate_depression() -> np.ndarray:
    """Faint circular cluster, some asymmetry, no eye."""
    img = _make_base(bg_val=10, noise_std=6)
    cx, cy = _random_center()
    # Central broad cloud mass
    img = _add_radial_gradient(img, cx, cy, radius=45, intensity=140)
    # Some scattered peripheral clouds
    for _ in range(np.random.randint(3, 7)):
        ox = cx + np.random.randint(-35, 36)
        oy = cy + np.random.randint(-35, 36)
        img = _add_radial_gradient(img, ox, oy, radius=15, intensity=60)
    return _pil_blur(img, radius=2.5)


def generate_cyclonic_storm() -> np.ndarray:
    """Defined spiral bands, no clear eye."""
    img = _make_base(bg_val=8, noise_std=5)
    cx, cy = _random_center()
    # CDO mass
    img = _add_radial_gradient(img, cx, cy, radius=40, intensity=180)
    # Spiral arms
    n_arms = np.random.choice([2, 3])
    img = _add_spiral_arms(img, cx, cy, num_arms=n_arms,
                           tightness=12, arm_width=6,
                           brightness=120, max_radius=55)
    return _pil_blur(img, radius=2.0)


def generate_severe_cyclonic_storm() -> np.ndarray:
    """Tight spiral, emerging eye structure."""
    img = _make_base(bg_val=5, noise_std=4)
    cx, cy = _random_center()
    # Dense CDO
    img = _add_radial_gradient(img, cx, cy, radius=35, intensity=210)
    # Tight spirals
    img = _add_spiral_arms(img, cx, cy, num_arms=3,
                           tightness=8, arm_width=5,
                           brightness=150, max_radius=50)
    # Emerging (partial) eye
    eye_r = np.random.uniform(3, 5)
    img = _add_eye(img, cx, cy, eye_radius=eye_r, darkness=25)
    return _pil_blur(img, radius=1.5)


def generate_very_severe_plus() -> np.ndarray:
    """Compact CDO, clear eye, symmetric spiral arms."""
    img = _make_base(bg_val=3, noise_std=3)
    cx, cy = _random_center()
    # Bright dense CDO
    img = _add_radial_gradient(img, cx, cy, radius=32, intensity=240)
    # Symmetric spiral arms
    img = _add_spiral_arms(img, cx, cy, num_arms=4,
                           tightness=6, arm_width=4,
                           brightness=180, max_radius=55)
    # Clear eye
    eye_r = np.random.uniform(5, 8)
    img = _add_eye(img, cx, cy, eye_radius=eye_r, darkness=10)
    return _pil_blur(img, radius=1.0)


# ──────────────────────────────────────────────────────────
# Generator dispatch
# ──────────────────────────────────────────────────────────
GENERATORS = [
    generate_no_cyclone,
    generate_depression,
    generate_cyclonic_storm,
    generate_severe_cyclonic_storm,
    generate_very_severe_plus
]


def generate_all():
    """Generate the full dataset and save to disk."""
    total = 0

    for class_idx, class_dir in enumerate(CLASS_DIRS):
        out_path = os.path.join(OUTPUT_DIR, class_dir)
        os.makedirs(out_path, exist_ok=True)

        gen_fn = GENERATORS[class_idx]
        print(f"  Generating class {class_idx}: {class_dir} ...")

        for i in range(IMAGES_PER_CLASS):
            arr = gen_fn()
            pil = Image.fromarray(arr.astype(np.uint8), mode="L")
            fname = os.path.join(out_path, f"img_{i:04d}.png")
            pil.save(fname)
            total += 1

    print(f"\n  ✅ Generated {total} images across {len(CLASS_DIRS)} classes.")
    print(f"  📂 Output directory: {OUTPUT_DIR}")
    return OUTPUT_DIR


if __name__ == "__main__":
    print("=" * 60)
    print("Cyclone AI — Synthetic IR Satellite Image Generator")
    print("=" * 60)
    generate_all()
