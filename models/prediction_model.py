# models/prediction_model.py

import os
import math
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# CNN FEATURE EXTRACTOR
# ============================================================

class SatelliteCNN(nn.Module):
    """
    CNN feature extractor for multi-source satellite imagery.

    Expected input:
        [batch, channels, height, width]

    Channels:
        0 -> IR
        1 -> WV
        2 -> VIS
        3 -> SST
    """

    def __init__(
        self,
        input_channels: int = 4,
        feature_size: int = 128
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                input_channels,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, feature_size),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x


# ============================================================
# CNN + LSTM MODEL
# ============================================================

class CycloneCNNLSTM(nn.Module):
    """
    Multi-source cyclone prediction model.

    Satellite input:
        [B, T, 4, H, W]

    Environmental input:
        [B, T, F]

    Output:
        track      -> latitude/longitude displacement
        intensity  -> wind and pressure evolution
        confidence -> prediction confidence
    """

    def __init__(
        self,
        satellite_channels: int = 4,
        environmental_features: int = 12,
        cnn_features: int = 128,
        hidden_size: int = 128,
        num_layers: int = 2
    ):
        super().__init__()

        self.cnn = SatelliteCNN(
            input_channels=satellite_channels,
            feature_size=cnn_features
        )

        self.lstm_input_size = (
            cnn_features +
            environmental_features
        )

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        # Track:
        # latitude displacement
        # longitude displacement
        self.track_head = nn.Linear(
            hidden_size,
            2
        )

        # Intensity:
        # wind
        # pressure
        self.intensity_head = nn.Linear(
            hidden_size,
            2
        )

        # Confidence
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        satellite,
        environmental
    ):

        # ----------------------------------------------------
        # satellite
        # [B,T,C,H,W]
        # ----------------------------------------------------

        batch_size = satellite.shape[0]
        sequence_length = satellite.shape[1]

        channels = satellite.shape[2]
        height = satellite.shape[3]
        width = satellite.shape[4]

        satellite = satellite.reshape(
            batch_size * sequence_length,
            channels,
            height,
            width
        )

        cnn_features = self.cnn(satellite)

        cnn_features = cnn_features.reshape(
            batch_size,
            sequence_length,
            -1
        )

        # ----------------------------------------------------
        # Feature fusion
        # ----------------------------------------------------

        fused = torch.cat(
            [
                cnn_features,
                environmental
            ],
            dim=-1
        )

        # ----------------------------------------------------
        # LSTM
        # ----------------------------------------------------

        lstm_output, _ = self.lstm(fused)

        last_output = lstm_output[:, -1, :]

        # ----------------------------------------------------
        # Prediction heads
        # ----------------------------------------------------

        track = self.track_head(last_output)

        intensity = self.intensity_head(last_output)

        confidence = self.confidence_head(last_output)

        return {
            "track": track,
            "intensity": intensity,
            "confidence": confidence
        }


# ============================================================
# CYCLONE PREDICTION MODEL
# ============================================================

class CyclonePredictionModel:
    """
    Main interface used by FastAPI.

    Responsibilities:

    1. Load trained PyTorch model
    2. Prepare input data
    3. Run prediction
    4. Generate cyclone trajectory
    5. Generate intensity forecast
    6. Generate confidence
    7. Generate uncertainty cone
    """

    FORECAST_HOURS = [
        6,
        12,
        24,
        48,
        72
    ]

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None
    ):

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        if device:
            self.device = torch.device(device)

        else:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.model_path = model_path

        self.model = None

        self.model_loaded = False

        self.model_mode = "fallback"

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        if model_path and os.path.exists(model_path):

            try:

                self.model = CycloneCNNLSTM(
                    satellite_channels=4,
                    environmental_features=12,
                    cnn_features=128,
                    hidden_size=128,
                    num_layers=2
                )

                checkpoint = torch.load(
                    model_path,
                    map_location=self.device
                )

                # ------------------------------------------------
                # Different checkpoint formats
                # ------------------------------------------------

                if isinstance(checkpoint, dict):

                    if "model_state_dict" in checkpoint:

                        state_dict = checkpoint[
                            "model_state_dict"
                        ]

                    elif "state_dict" in checkpoint:

                        state_dict = checkpoint[
                            "state_dict"
                        ]

                    else:

                        state_dict = checkpoint

                else:

                    state_dict = checkpoint

                self.model.load_state_dict(
                    state_dict,
                    strict=False
                )

                self.model.to(self.device)

                self.model.eval()

                self.model_loaded = True

                self.model_mode = "pytorch_cnn_lstm"

                print(
                    f"[INFO] CNN-LSTM model loaded: "
                    f"{model_path}"
                )

            except Exception as exc:

                print(
                    "[WARNING] Failed to load "
                    f"CNN-LSTM model: {exc}"
                )

                self.model = None

                self.model_loaded = False

                self.model_mode = "fallback"

        else:

            print(
                "[WARNING] CNN-LSTM checkpoint "
                "not found."
            )

            print(
                "[INFO] Running fallback prediction mode."
            )


    # ========================================================
    # NORMALIZATION
    # ========================================================

    @staticmethod
    def normalize_satellite(
        satellite: np.ndarray
    ) -> np.ndarray:
        """
        Normalize satellite image.

        Input:
            arbitrary satellite values

        Output:
            approximately [-1,1]
        """

        satellite = np.nan_to_num(
            satellite,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        satellite = satellite.astype(
            np.float32
        )

        # Per-channel normalization
        for channel in range(
            satellite.shape[0]
        ):

            data = satellite[channel]

            mean = np.mean(data)

            std = np.std(data)

            if std < 1e-6:

                std = 1.0

            satellite[channel] = (
                data - mean
            ) / std

        return satellite


    # ========================================================
    # PREPARE SATELLITE INPUT
    # ========================================================

    def prepare_satellite_input(
        self,
        satellite_data: Optional[np.ndarray] = None
    ) -> torch.Tensor:
        """
        Prepare satellite sequence.

        Expected shape:

            [T,4,H,W]

        If data is not provided, a zero tensor
        is created for API testing.
        """

        if satellite_data is None:

            satellite_data = np.zeros(
                (3, 4, 224, 224),
                dtype=np.float32
            )

        satellite_data = np.asarray(
            satellite_data,
            dtype=np.float32
        )

        if satellite_data.ndim != 4:

            raise ValueError(
                "Satellite data must have shape "
                "[T,4,H,W]"
            )

        if satellite_data.shape[1] != 4:

            raise ValueError(
                "Satellite input must contain "
                "4 channels: IR, WV, VIS, SST"
            )

        normalized = []

        for frame in satellite_data:

            normalized.append(
                self.normalize_satellite(
                    frame
                )
            )

        satellite_data = np.stack(
            normalized
        )

        tensor = torch.tensor(
            satellite_data,
            dtype=torch.float32
        )

        # [T,C,H,W]
        # -> [B,T,C,H,W]

        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)


    # ========================================================
    # PREPARE ENVIRONMENTAL INPUT
    # ========================================================

    def prepare_environmental_input(
        self,
        environmental_data: Optional[np.ndarray] = None
    ) -> torch.Tensor:
        """
        Environmental features:

        0  -> latitude
        1  -> longitude
        2  -> wind
        3  -> pressure
        4  -> SST
        5  -> humidity
        6  -> vertical wind shear
        7  -> steering speed
        8  -> steering direction sin
        9  -> steering direction cos
        10 -> ocean heat content
        11 -> forecast hour
        """

        if environmental_data is None:

            environmental_data = np.zeros(
                (3, 12),
                dtype=np.float32
            )

        environmental_data = np.asarray(
            environmental_data,
            dtype=np.float32
        )

        if environmental_data.ndim != 2:

            raise ValueError(
                "Environmental data must have "
                "shape [T,12]"
            )

        if environmental_data.shape[1] != 12:

            raise ValueError(
                "Environmental input must contain "
                "12 features."
            )

        environmental_data = np.nan_to_num(
            environmental_data,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        tensor = torch.tensor(
            environmental_data,
            dtype=torch.float32
        )

        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)


    # ========================================================
    # CNN-LSTM PREDICTION
    # ========================================================

    def predict_cnn_lstm(
        self,
        satellite_data: Optional[np.ndarray] = None,
        environmental_data: Optional[np.ndarray] = None,
        current_lat: float = 14.5,
        current_lon: float = 86.2,
        current_wind: float = 55.0,
        current_pressure: float = 988.0
    ) -> Dict[str, Any]:

        # ----------------------------------------------------
        # If trained model exists
        # ----------------------------------------------------

        if self.model_loaded and self.model is not None:

            try:

                satellite = (
                    self.prepare_satellite_input(
                        satellite_data
                    )
                )

                environmental = (
                    self.prepare_environmental_input(
                        environmental_data
                    )
                )

                with torch.no_grad():

                    output = self.model(
                        satellite,
                        environmental
                    )

                track = (
                    output["track"]
                    .cpu()
                    .numpy()[0]
                )

                intensity = (
                    output["intensity"]
                    .cpu()
                    .numpy()[0]
                )

                confidence = float(
                    output["confidence"]
                    .cpu()
                    .numpy()[0][0]
                )

                return self._build_forecast(
                    track,
                    intensity,
                    confidence,
                    current_lat,
                    current_lon,
                    current_wind,
                    current_pressure
                )

            except Exception as exc:

                print(
                    "[WARNING] CNN-LSTM prediction "
                    f"failed: {exc}"
                )

        # ----------------------------------------------------
        # Fallback
        # ----------------------------------------------------

        return self._fallback_prediction(
            current_lat=current_lat,
            current_lon=current_lon,
            current_wind=current_wind,
            current_pressure=current_pressure
        )


    # ========================================================
    # BUILD FORECAST
    # ========================================================

    def _build_forecast(
        self,
        track,
        intensity,
        confidence,
        current_lat,
        current_lon,
        current_wind,
        current_pressure
    ):

        # Model outputs are treated as displacement.
        lat_step = float(track[0])

        lon_step = float(track[1])

        predicted_wind_change = float(
            intensity[0]
        )

        predicted_pressure_change = float(
            intensity[1]
        )

        # ----------------------------------------------------
        # Safety limits
        # ----------------------------------------------------

        lat_step = np.clip(
            lat_step,
            -2.0,
            2.0
        )

        lon_step = np.clip(
            lon_step,
            -3.0,
            3.0
        )

        predicted_wind_change = np.clip(
            predicted_wind_change,
            -30.0,
            30.0
        )

        predicted_pressure_change = np.clip(
            predicted_pressure_change,
            -50.0,
            50.0
        )

        confidence = float(
            np.clip(
                confidence,
                0.05,
                0.99
            )
        )

        trajectory = []

        # Current storm position
        previous_lat = current_lat
        previous_lon = current_lon

        for index, hours in enumerate(
            self.FORECAST_HOURS
        ):

            fraction = hours / 72.0

            lat = (
                current_lat +
                lat_step * fraction
            )

            lon = (
                current_lon +
                lon_step * fraction
            )

            # Intensity evolution
            wind = (
                current_wind +
                predicted_wind_change *
                fraction
            )

            pressure = (
                current_pressure +
                predicted_pressure_change *
                fraction
            )

            # Keep physically reasonable values
            wind = float(
                np.clip(
                    wind,
                    15,
                    200
                )
            )

            pressure = float(
                np.clip(
                    pressure,
                    850,
                    1030
                )
            )

            point_confidence = (
                confidence *
                (1.0 - 0.08 * index)
            )

            point_confidence = float(
                np.clip(
                    point_confidence,
                    0.05,
                    0.99
                )
            )

            trajectory.append(
                {
                    "forecast_hour": hours,
                    "latitude": round(
                        float(lat),
                        4
                    ),
                    "longitude": round(
                        float(lon),
                        4
                    ),
                    "wind_knots": round(
                        wind,
                        2
                    ),
                    "mslp_hpa": round(
                        pressure,
                        2
                    ),
                    "confidence": round(
                        point_confidence,
                        4
                    )
                }
            )

            previous_lat = lat
            previous_lon = lon

        uncertainty = (
            self._generate_uncertainty_polygon(
                trajectory,
                confidence
            )
        )

        return {
            "success": True,

            "model": {
                "name": "CycloneCNNLSTM",
                "mode": self.model_mode,
                "loaded": self.model_loaded,
                "device": str(self.device)
            },

            "current_position": {
                "latitude": current_lat,
                "longitude": current_lon,
                "wind_knots": current_wind,
                "mslp_hpa": current_pressure
            },

            "trajectory_points": trajectory,

            "confidence": round(
                confidence,
                4
            ),

            "confidence_percent": round(
                confidence * 100,
                2
            ),

            "uncertainty_polygon": uncertainty,

            "warning": None
        }


    # ========================================================
    # FALLBACK MODEL
    # ========================================================

    def _fallback_prediction(
        self,
        current_lat: float,
        current_lon: float,
        current_wind: float,
        current_pressure: float
    ) -> Dict[str, Any]:
        """
        Used when trained checkpoint is unavailable.

        This is NOT an AI prediction.

        It provides a safe demonstration trajectory
        so that the frontend remains functional.
        """

        trajectory = []

        # Demo movement toward northeast
        lat_direction = 0.025

        lon_direction = 0.045

        for index, hours in enumerate(
            self.FORECAST_HOURS
        ):

            lat = (
                current_lat +
                lat_direction * hours
            )

            lon = (
                current_lon +
                lon_direction * hours
            )

            # Small weakening for demonstration
            wind = (
                current_wind -
                0.05 * hours
            )

            pressure = (
                current_pressure +
                0.10 * hours
            )

            wind = max(
                15,
                wind
            )

            trajectory.append(
                {
                    "forecast_hour": hours,
                    "latitude": round(
                        lat,
                        4
                    ),
                    "longitude": round(
                        lon,
                        4
                    ),
                    "wind_knots": round(
                        wind,
                        2
                    ),
                    "mslp_hpa": round(
                        pressure,
                        2
                    ),
                    "confidence": round(
                        max(
                            0.35,
                            0.85 -
                            index * 0.10
                        ),
                        4
                    )
                }
            )

        uncertainty = (
            self._generate_uncertainty_polygon(
                trajectory,
                0.60
            )
        )

        return {
            "success": True,

            "model": {
                "name": "CycloneCNNLSTM",
                "mode": "fallback",
                "loaded": False,
                "device": str(self.device)
            },

            "current_position": {
                "latitude": current_lat,
                "longitude": current_lon,
                "wind_knots": current_wind,
                "mslp_hpa": current_pressure
            },

            "trajectory_points": trajectory,

            "confidence": 0.60,

            "confidence_percent": 60.0,

            "uncertainty_polygon": uncertainty,

            "warning": (
                "Trained CNN-LSTM checkpoint was not "
                "available. Fallback demonstration "
                "trajectory is being used."
            )
        }


    # ========================================================
    # UNCERTAINTY CONE
    # ========================================================

    @staticmethod
    def _generate_uncertainty_polygon(
        trajectory: List[Dict[str, Any]],
        confidence: float
    ) -> List[List[float]]:
        """
        Generate a widening uncertainty polygon.

        Returns Leaflet-compatible:

            [[lat, lon], ...]
        """

        left_side = []

        right_side = []

        confidence = float(
            np.clip(
                confidence,
                0.05,
                0.99
            )
        )

        for index, point in enumerate(
            trajectory
        ):

            lat = point["latitude"]

            lon = point["longitude"]

            hours = point["forecast_hour"]

            # Cone widens with forecast time
            base_width = (
                0.10 +
                hours * 0.025
            )

            # Lower confidence = wider cone
            width = (
                base_width *
                (1.5 - confidence)
            )

            left_side.append(
                [
                    round(
                        lat + width,
                        4
                    ),
                    round(
                        lon - width,
                        4
                    )
                ]
            )

            right_side.append(
                [
                    round(
                        lat - width,
                        4
                    ),
                    round(
                        lon + width,
                        4
                    )
                ]
            )

        polygon = (
            left_side +
            right_side[::-1]
        )

        return polygon


    # ========================================================
    # OLD API COMPATIBILITY
    # ========================================================

    def predict_track_and_intensity(
        self,
        current_lat: float = 14.5,
        current_lon: float = 86.2,
        current_wind: float = 55.0,
        current_pressure: float = 988.0,
        satellite_data: Optional[np.ndarray] = None,
        environmental_data: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:

        return self.predict_cnn_lstm(
            satellite_data=satellite_data,
            environmental_data=environmental_data,
            current_lat=current_lat,
            current_lon=current_lon,
            current_wind=current_wind,
            current_pressure=current_pressure
        )


    # Alias
    predict = predict_track_and_intensity


# ============================================================
# MODEL INITIALIZATION
# ============================================================

def create_prediction_model():
    """
    Automatically locate the trained checkpoint.
    """

    current_file = os.path.abspath(
        __file__
    )

    project_root = os.path.abspath(
        os.path.join(
            os.path.dirname(current_file),
            ".."
        )
    )

    model_path = os.path.join(
        project_root,
        "models",
        "cyclone_trajectory_lstm.pth"
    )

    # Optional CNN-LSTM checkpoint
    cnn_lstm_path = os.path.join(
        project_root,
        "models",
        "cyclone_cnn_lstm.pth"
    )

    if os.path.exists(
        cnn_lstm_path
    ):

        model_path = cnn_lstm_path

    print("=" * 60)
    print("CYCLONE AI PREDICTION MODEL")
    print("=" * 60)

    print(
        f"Model path: {model_path}"
    )

    print(
        f"Checkpoint exists: "
        f"{os.path.exists(model_path)}"
    )

    model = CyclonePredictionModel(
        model_path=(
            model_path
            if os.path.exists(model_path)
            else None
        )
    )

    print(
        f"Model loaded: "
        f"{model.model_loaded}"
    )

    print(
        f"Model mode: "
        f"{model.model_mode}"
    )

    print(
        f"Device: "
        f"{model.device}"
    )

    print("=" * 60)

    return model


# ============================================================
# GLOBAL MODEL INSTANCE
# ============================================================

prediction_model = create_prediction_model()


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    print("\nRunning local prediction test...\n")

    result = (
        prediction_model.predict_track_and_intensity(
            current_lat=14.5,
            current_lon=86.2,
            current_wind=55,
            current_pressure=988
        )
    )

    print("\nPrediction Result:")
    print("-" * 60)

    print(
        "Model:",
        result["model"]
    )

    print(
        "Confidence:",
        result["confidence_percent"],
        "%"
    )

    print("\nTrajectory:")

    for point in result[
        "trajectory_points"
    ]:

        print(
            f"+{point['forecast_hour']}h | "
            f"Lat={point['latitude']} | "
            f"Lon={point['longitude']} | "
            f"Wind={point['wind_knots']} kt | "
            f"MSLP={point['mslp_hpa']} hPa"
        )

    print("\nUncertainty points:")

    for point in result[
        "uncertainty_polygon"
    ][:5]:

        print(point)